"""Class definitions for corpora"""
from __future__ import annotations

import collections
import logging
import os
import re
import threading
import time
import typing
from abc import ABCMeta, abstractmethod
from pathlib import Path

import polars as pl

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import DatabaseMixin, MfaWorker
from montreal_forced_aligner.corpus.classes import FileData, UtteranceData
from montreal_forced_aligner.corpus.multiprocessing import (
    ExportKaldiFilesArguments,
    ExportKaldiFilesFunction,
    NormalizeTextFunction,
    dictionary_ids_for_job,
)
from montreal_forced_aligner.data import (
    DatabaseImportData,
    Language,
    TextFileType,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.db_polars import (
    Corpus,
    CorpusWorkflow,
    Dialect,
    Dictionary,
    Dictionary2Job,
    File,
    Job,
    Pronunciation,
    SoundFile,
    Speaker,
    SpeakerOrdering,
    TextFile,
    Utterance,
    Word,
)
from montreal_forced_aligner.exceptions import CorpusError
from montreal_forced_aligner.helper import mfa_open, output_mapping
from montreal_forced_aligner.utils import run_kaldi_function

__all__ = ["CorpusMixin"]

logger = logging.getLogger("mfa")


class CorpusMixin(MfaWorker, DatabaseMixin, metaclass=ABCMeta):
    """
    Mixin class for processing corpora

    Notes
    -----
    Using characters in files to specify speakers is generally finicky and leads to errors, so I would not
    recommend using it.  Additionally, consider it deprecated and could be removed in future versions

    Parameters
    ----------
    corpus_directory: str
        Path to corpus
    speaker_characters: int or str, optional
        Number of characters in the file name to specify the speaker
    ignore_speakers: bool
        Flag for whether to discard any parsed speaker information during top-level worker's processing
    oov_count_threshold: int
        Words in the corpus with counts less than or equal to the threshold will be treated as OOV items, defaults to 0

    See Also
    --------
    :class:`~montreal_forced_aligner.abc.MfaWorker`
        For MFA processing parameters
    :class:`~montreal_forced_aligner.abc.TemporaryDirectoryMixin`
        For temporary directory parameters

    Attributes
    ----------
    jobs: list[:class:`~montreal_forced_aligner.corpus.multiprocessing.Job`]
        List of jobs for processing the corpus and splitting speakers
    stopped: :class:`~threading.Event`
        Stop check for loading the corpus
    decode_error_files: list[str]
        List of text files that could not be loaded with utf8
    textgrid_read_errors: list[str]
        List of TextGrid files that had an error in loading
    """

    def __init__(
        self,
        corpus_directory: str,
        speaker_characters: typing.Union[int, str] = 0,
        ignore_speakers: bool = False,
        oov_count_threshold: int = 0,
        language: Language = Language.unknown,
        **kwargs,
    ):
        if not os.path.exists(corpus_directory):
            raise CorpusError(f"The directory '{corpus_directory}' does not exist.")
        if not os.path.isdir(corpus_directory):
            raise CorpusError(
                f"The specified path for the corpus ({corpus_directory}) is not a directory."
            )
        self._speaker_ids = {}
        self.corpus_directory = corpus_directory
        self.speaker_characters = speaker_characters
        self.ignore_speakers = ignore_speakers
        self.oov_count_threshold = oov_count_threshold
        self.stopped = threading.Event()
        self.decode_error_files = []
        self.textgrid_read_errors = []
        self._num_speakers = None
        self._num_utterances = None
        self._num_files = None
        super().__init__(**kwargs)
        os.makedirs(self.corpus_output_directory, exist_ok=True)
        self.imported = False
        self.text_normalized = False
        self._current_speaker_index = 1
        self._current_file_index = 1
        self._current_utterance_index = 1
        self._speaker_ids = {}
        self._word_set = []
        self._jobs = []
        self.ignore_empty_utterances = False
        self.language = language
        if isinstance(language, str):
            self.language = Language[language.split(".")[-1]]

    # @property
    # def jobs(self) -> typing.List[Job]:
    #     if not self._jobs:
    #         with self.session() as session:
    #             c: Corpus = session.query(Corpus).first()
    #             jobs = session.query(Job).options(
    #                 joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries)
    #             )
    #             if c.current_subset:
    #                 jobs = jobs.filter(Job.utterances.any(Utterance.in_subset == True))  # noqa
    #             jobs = jobs.filter(Job.utterances.any(Utterance.ignored == False))  # noqa
    #             self._jobs = jobs.all()
    #     return self._jobs

    # def dictionary_ids_for_job(self, job_id):
    #     with self.session() as session:
    #         return dictionary_ids_for_job(session, job_id)

    # def inspect_database(self) -> None:
    #     """Check if a database file exists and create the necessary metadata"""
    #     self.initialize_database()
    #     with self.session() as session:
    #         corpus = session.query(Corpus).first()
    #         if corpus:
    #             self.imported = corpus.imported
    #             self.text_normalized = corpus.text_normalized
    #         else:
    #             session.add(
    #                 Corpus(
    #                     name=self.data_source_identifier,
    #                     path=self.corpus_directory,
    #                     data_directory=self.corpus_output_directory,
    #                 )
    #             )
    #             session.commit()

    @property
    def jobs(self) -> typing.List[Job]:
        """
        Retrieve the list of job entries from the Polars DB.
        Instead of using a SQLAlchemy session, we pull the corresponding tables from the in-memory DB.
        Note: This implementation assumes that each job record contains a key "utterances" which is a list of
        dictionaries (each representing an utterance) having boolean keys "in_subset" and "ignored".
        """
        if not self._jobs:
            # Get the corpus record from the Polars DB (if any)
            corpus_df = self.polars_db.get_table("corpus")
            corpus = None
            if not corpus_df.is_empty():
                corpus_data = corpus_df.to_dicts()[0]
                corpus = Corpus.from_dict(corpus_data)
            
            # Get all jobs from the Polars DB
            jobs_df = self.polars_db.get_table("job")
            job_dicts = jobs_df.to_dicts() if not jobs_df.is_empty() else []
    
            def job_filter(job: dict) -> bool:
                # Retrieve the utterances list; if empty, the job will be filtered out
                utterances = job.get("utterances", [])
    
                # If the corpus has a nonzero current_subset value,
                # require that at least one utterance has "in_subset" set to True.
                if corpus and getattr(corpus, "current_subset", 0):
                    if not any(u.get("in_subset", False) for u in utterances):
                        return False
    
                # Only include jobs where at least one utterance is not ignored.
                if not any(not u.get("ignored", True) for u in utterances):
                    return False
    
                return True
    
            # Convert filtered dictionaries back to Job instances.
            filtered_jobs = [Job.from_dict(j) for j in job_dicts if job_filter(j)]
            self._jobs = filtered_jobs
        return self._jobs

    def dictionary_ids_for_job(self, job_id):
        """
        Find dictionary ids for the job with the given id.
        It assumes that the job record has a "dictionaries" key that is a list of dictionaries,
        each with an "id" field.
        """
        jobs_df = self.polars_db.get_table("job")
        if not jobs_df.is_empty():
            job_dicts = jobs_df.to_dicts()
            # Find the job record matching the given job_id
            job_record = next((job for job in job_dicts if job.get("id") == job_id), None)
            if job_record and "dictionaries" in job_record:
                return [d.get("id") for d in job_record["dictionaries"] if d.get("id") is not None]
        return []

    def inspect_database(self) -> None:
        """
        Check if a corpus record exists in the Polars DB and, if so, retrieve its properties.
        Otherwise create a new corpus record.
        """
        self.initialize_database()  # Ensure that self.db is created and tables exist

        corpus_df = self.polars_db.get_table("corpus")
        if not corpus_df.is_empty():
            corpus_data = corpus_df.to_dicts()[0]
            corpus = Corpus.from_dict(corpus_data)
            self.imported = corpus.imported
            self.text_normalized = corpus.text_normalized
        else:
            # If no corpus record exists, add a new one
            new_corpus = Corpus(
                name=self.data_source_identifier,
                path=self.corpus_directory,
                data_directory=self.corpus_output_directory,
            )
            self.polars_db.add_row("corpus", new_corpus.to_dict())

    # def get_utterances(
    #     self,
    #     id: typing.Optional[int] = None,
    #     file: typing.Optional[typing.Union[str, int]] = None,
    #     speaker: typing.Optional[typing.Union[str, int]] = None,
    #     begin: typing.Optional[float] = None,
    #     end: typing.Optional[float] = None,
    #     session: Session = None,
    # ):
    #     """
    #     Get a file from search parameters

    #     Parameters
    #     ----------
    #     id: int
    #         Integer ID to look up
    #     file: str or int
    #         File name or ID to look up
    #     speaker: str or int
    #         Speaker name or ID to look up
    #     begin: float
    #         Begin timestamp to look up
    #     end: float
    #         Ending timestamp to look up

    #     Returns
    #     -------
    #     :class:`~montreal_forced_aligner.db.Utterance`
    #         Utterance match
    #     """
    #     if session is None:
    #         session = self.session()
    #     if id is not None:
    #         utterance = session.get(Utterance, id)
    #         if not utterance:
    #             raise Exception(f"Could not find utterance with id of {id}")
    #         return utterance
    #     else:
    #         utterance = session.query(Utterance)
    #         if file is not None:
    #             utterance = utterance.join(Utterance.file)
    #             if isinstance(file, int):
    #                 utterance = utterance.filter(File.id == file)
    #             else:
    #                 utterance = utterance.filter(File.name == file)
    #         if speaker is not None:
    #             utterance = utterance.join(Utterance.speaker)
    #             if isinstance(speaker, int):
    #                 utterance = utterance.filter(Speaker.id == speaker)
    #             else:
    #                 utterance = utterance.filter(Speaker.name == speaker)
    #         if begin is not None:
    #             utterance = utterance.filter(Utterance.begin == begin)
    #         if end is not None:
    #             utterance = utterance.filter(Utterance.end == end)
    #         utterance = utterance.all()
    #         return list(utterance)
    def get_utterances(
        self,
        id: typing.Optional[int] = None,
        file: typing.Optional[typing.Union[str, int]] = None,
        speaker: typing.Optional[typing.Union[str, int]] = None,
        begin: typing.Optional[float] = None,
        end: typing.Optional[float] = None,
    ):
        """
        Get utterances based on the provided search parameters using the in-memory Polars DB.

        Parameters
        ----------
        id : int, optional
            Integer ID to look up.
        file : str or int, optional
            File name or ID to look up.
        speaker : str or int, optional
            Speaker name or ID to look up.
        begin : float, optional
            Begin timestamp to filter on.
        end : float, optional
            Ending timestamp to filter on.

        Returns
        -------
        Utterance or list[Utterance]
            If an ID is specified, returns a single Utterance instance.
            Otherwise, returns a list of matching Utterance instances.
        """

        # Get the utterance table from the Polars database
        utterance_df = self.polars_db.get_table("utterance")

        # If an ID is provided, filter by the primary key (id)
        if id is not None:
            filtered = utterance_df.filter(pl.col("id") == id)
            if filtered.is_empty():
                raise Exception(f"Could not find utterance with id of {id}")
            u_dict = filtered.to_dicts()[0]
            return Utterance.from_dict(u_dict)
        else:
            filtered = utterance_df

            # Filter based on file information if provided
            if file is not None:
                if isinstance(file, int):
                    filtered = filtered.filter(pl.col("file_id") == file)
                else:
                    # When file is specified by name, look up corresponding file ids
                    file_df = self.polars_db.get_table("file")
                    file_ids = (
                        file_df.filter(pl.col("name") == file)["id"].to_list()
                        if not file_df.is_empty()
                        else []
                    )
                    if not file_ids:
                        # No matching file found; produce an empty DataFrame
                        filtered = filtered.filter(pl.lit(False))
                    else:
                        filtered = filtered.filter(pl.col("file_id").is_in(file_ids))

            # Filter based on speaker information if provided
            if speaker is not None:
                if isinstance(speaker, int):
                    filtered = filtered.filter(pl.col("speaker_id") == speaker)
                else:
                    speaker_df = self.polars_db.get_table("speaker")
                    speaker_ids = (
                        speaker_df.filter(pl.col("name") == speaker)["id"].to_list()
                        if not speaker_df.is_empty()
                        else []
                    )
                    if not speaker_ids:
                        filtered = filtered.filter(pl.lit(False))
                    else:
                        filtered = filtered.filter(pl.col("speaker_id").is_in(speaker_ids))

            # Filter by begin timestamp if specified
            if begin is not None:
                filtered = filtered.filter(pl.col("begin") == begin)

            # Filter by end timestamp if specified
            if end is not None:
                filtered = filtered.filter(pl.col("end") == end)

            # Convert the resulting rows into a list of Utterance objects
            result_dicts = filtered.to_dicts() if not filtered.is_empty() else []
            return [Utterance.from_dict(u) for u in result_dicts]

    # def get_file(
    #     self, id: typing.Optional[int] = None, name=None, session: Session = None
    # ) -> File:
    #     """
    #     Get a file from search parameters

    #     Parameters
    #     ----------
    #     id: int
    #         Integer ID to look up
    #     name: str
    #         File name to look up

    #     Returns
    #     -------
    #     :class:`~montreal_forced_aligner.db.File`
    #         File match
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True
    #     file = session.query(File).options(
    #         selectinload(File.utterances).joinedload(Utterance.speaker, innerjoin=True),
    #         joinedload(File.sound_file, innerjoin=True),
    #         joinedload(File.text_file, innerjoin=True),
    #         selectinload(File.speakers),
    #     )
    #     if id is not None:
    #         file = file.get(id)
    #         if not file:
    #             raise Exception(f"Could not find utterance with id of {id}")
    #         if close:
    #             session.close()
    #         return file
    #     else:
    #         file = file.filter(File.name == name).first()
    #         if not file:
    #             raise Exception(f"Could not find utterance with name of {name}")
    #         if close:
    #             session.close()
    #         return file
    def get_file(self, id: typing.Optional[int] = None, name: typing.Optional[str] = None) -> File:
        """
        Get a file from search parameters using the in-memory Polars DB.

        Parameters
        ----------
        id : int, optional
            Integer ID to look up.
        name : str, optional
            File name to look up.

        Returns
        -------
        File
            Matching File record.

        Raises
        ------
        Exception
            If no matching file is found or if neither id nor name is provided.
        """

        file_df = self.polars_db.get_table("file")
        if id is not None:
            filtered = file_df.filter(pl.col("id") == id)
            if filtered.is_empty():
                raise Exception(f"Could not find file with id of {id}")
            file_dict = filtered.to_dicts()[0]
            return File.from_dict(file_dict)
        elif name is not None:
            filtered = file_df.filter(pl.col("name") == name)
            if filtered.is_empty():
                raise Exception(f"Could not find file with name of {name}")
            file_dict = filtered.to_dicts()[0]
            return File.from_dict(file_dict)
        else:
            raise Exception("Must provide either an 'id' or 'name' to get a file")

    @property
    def corpus_meta(self) -> typing.Dict[str, typing.Any]:
        """Corpus metadata"""
        return {}

    @property
    def features_log_directory(self) -> Path:
        """Feature log directory"""
        return self.split_directory.joinpath("log")

    @property
    def split_directory(self) -> Path:
        """Directory used to store information split by job"""
        return self.corpus_output_directory.joinpath(f"split{config.NUM_JOBS}")

    # def _write_spk2utt(self) -> None:
    #     """Write spk2utt scp file for Kaldi"""
    #     data = {}
    #     utt2spk_data = {}
    #     with self.session() as session:
    #         utterances = (
    #             session.query(Utterance.kaldi_id, Utterance.speaker_id)
    #             .join(Utterance.speaker)
    #             .filter(Speaker.name != "MFA_UNKNOWN")
    #             .order_by(Utterance.kaldi_id)
    #         )

    #         for utt_id, speaker_id in utterances:
    #             if speaker_id not in data:
    #                 data[speaker_id] = []
    #             data[speaker_id].append(utt_id)
    #             utt2spk_data[utt_id] = speaker_id

    #     output_mapping(utt2spk_data, self.corpus_output_directory.joinpath("utt2spk.scp"))
    #     output_mapping(data, self.corpus_output_directory.joinpath("spk2utt.scp"))

    # def create_corpus_split(self) -> None:
    #     """Create split directory and output information from Jobs"""
    #     os.makedirs(self.split_directory.joinpath("log"), exist_ok=True)
    #     with self.session() as session:
    #         jobs = session.query(Job)
    #         arguments = [
    #             ExportKaldiFilesArguments(
    #                 j.id,
    #                 getattr(self, "session" if config.USE_THREADING else "db_string", ""),
    #                 None,
    #                 self.split_directory,
    #             )
    #             for j in jobs
    #         ]

    #     for _ in run_kaldi_function(
    #         ExportKaldiFilesFunction, arguments, total_count=self.num_utterances
    #     ):
    #         pass

    # @property
    # def corpus_word_set(self) -> typing.List[str]:
    #     """Set of words used in the corpus"""
    #     if not self._word_set:
    #         with self.session() as session:
    #             self._word_set = [
    #                 x[0]
    #                 for x in session.query(Word.word).filter(Word.count > 0).order_by(Word.word)
    #             ]
    #     return self._word_set
    def _write_spk2utt(self) -> None:
        """Write spk2utt scp file for Kaldi using the Polars in-memory DB."""

        data = {}
        utt2spk_data = {}
        # Get the utterance and speaker tables from the Polars DB
        utterance_df = self.polars_db.get_table("utterance")
        speaker_df = self.polars_db.get_table("speaker")
        # Join on the speaker id to get speaker details (assumes speaker table has a column "id" and "name")
        joined_df = utterance_df.join(speaker_df, left_on="speaker_id", right_on="id", how="inner")
        # Filter out speakers named "MFA_UNKNOWN" and sort by the Kaldi ID (assumes column "kaldi_id")
        filtered_df = joined_df.filter(pl.col("name") != "MFA_UNKNOWN").sort("kaldi_id")

        for row in filtered_df.to_dicts():
            utt_id = row.get("kaldi_id")
            speaker_id = row.get("speaker_id")
            if speaker_id not in data:
                data[speaker_id] = []
            data[speaker_id].append(utt_id)
            utt2spk_data[utt_id] = speaker_id

        output_mapping(utt2spk_data, self.corpus_output_directory.joinpath("utt2spk.scp"))
        output_mapping(data, self.corpus_output_directory.joinpath("spk2utt.scp"))

    def create_corpus_split(self) -> None:
        """Create split directory and output information from Jobs using the Polars in-memory DB."""
        import os
        os.makedirs(self.split_directory.joinpath("log"), exist_ok=True)
        # Retrieve all jobs from the Polars DB as dictionaries.
        job_df = self.polars_db.get_table("job")
        job_dicts = job_df.to_dicts() if not job_df.is_empty() else []
        arguments = [
            ExportKaldiFilesArguments(
                j.get("id"),
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                None,
                self.split_directory,
            )
            for j in job_dicts
        ]
        # Execute the Kaldi export function. This loop consumes the iterator returned by run_kaldi_function.
        for _ in run_kaldi_function(ExportKaldiFilesFunction, arguments, total_count=self.num_utterances):
            pass

    @property
    def corpus_word_set(self) -> typing.List[str]:
        """Set of words used in the corpus obtained from the in-memory DB."""
        if not self._word_set:
            word_df = self.polars_db.get_table("word")
            # Filter words having a positive count and sort by the word field.
            filtered_df = word_df.filter(pl.col("count") > 0).sort("word")
            self._word_set = [row["word"] for row in filtered_df.to_dicts()]
        return self._word_set


    # def add_utterance(self, utterance: UtteranceData, session: Session = None) -> Utterance:
    #     """
    #     Add an utterance to the corpus

    #     Parameters
    #     ----------
    #     utterance: :class:`~montreal_forced_aligner.corpus.classes.UtteranceData`
    #         Utterance to add
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True

    #     speaker_obj = session.query(Speaker).filter_by(name=utterance.speaker_name).first()
    #     if not speaker_obj:
    #         dictionary = None
    #         if hasattr(self, "get_dictionary"):
    #             dictionary = (
    #                 session.query(Dictionary)
    #                 .filter_by(name=self.get_dictionary(utterance.speaker_name).name)
    #                 .first()
    #             )
    #         speaker_obj = Speaker(name=utterance.speaker_name, dictionary=dictionary)
    #         session.add(speaker_obj)
    #         self._speaker_ids[utterance.speaker_name] = speaker_obj
    #     else:
    #         self._speaker_ids[utterance.speaker_name] = speaker_obj
    #     file_obj = session.query(File).filter_by(name=utterance.file_name).first()
    #     u = Utterance.from_data(
    #         utterance, file_obj, speaker_obj, frame_shift=getattr(self, "frame_shift", None)
    #     )
    #     u.id = self.get_next_primary_key(Utterance)
    #     session.add(u)
    #     if close:
    #         session.commit()
    #         session.close()
    #     return u

    # def delete_utterance(self, utterance_id: int, session: Session = None) -> None:
    #     """
    #     Delete an utterance from the corpus

    #     Parameters
    #     ----------
    #     utterance_id: int
    #         Utterance to delete
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True

    #     session.query(Utterance).filter(Utterance.id == utterance_id).delete()
    #     session.commit()
    #     if close:
    #         session.close()
    def add_utterance(self, utterance: UtteranceData) -> Utterance:
        """
        Add an utterance to the corpus using the in-memory Polars DB.

        Parameters
        ----------
        utterance : UtteranceData
            Utterance data to add.

        Returns
        -------
        Utterance
            The created utterance instance.
        """

        # Look up the speaker by name in the Polars DB.
        speaker_df = self.polars_db.get_table("speaker")
        filtered_speaker = speaker_df.filter(pl.col("name") == utterance.speaker_name)
        if not filtered_speaker.is_empty():
            speaker_dict = filtered_speaker.to_dicts()[0]
            speaker_obj = Speaker.from_dict(speaker_dict)
        else:
            # Obtain dictionary if available.
            dictionary = None
            if hasattr(self, "get_dictionary"):
                # Look up the dictionary based on the returned dictionary name.
                dict_name = self.get_dictionary(utterance.speaker_name).name
                dict_df = self.polars_db.get_table("dictionary").filter(pl.col("name") == dict_name)
                if not dict_df.is_empty():
                    dictionary = Dictionary.from_dict(dict_df.to_dicts()[0])
            speaker_obj = Speaker(name=utterance.speaker_name, dictionary=dictionary)
            self.polars_db.add_row("speaker", speaker_obj.to_dict())

        self._speaker_ids[utterance.speaker_name] = speaker_obj

        # Look up the file record by file name.
        file_df = self.polars_db.get_table("file").filter(pl.col("name") == utterance.file_name)
        if file_df.is_empty():
            raise Exception(f"Could not find file with name of {utterance.file_name}")
        file_obj = File.from_dict(file_df.to_dicts()[0])

        # Create the Utterance object from the given data.
        u = Utterance.from_data(
            utterance,
            file_obj,
            speaker_obj,
            frame_shift=getattr(self, "frame_shift", None),
        )
        # Assign a new primary key to the utterance.
        u.id = self.polars_db.get_next_primary_key("utterance")
        self.polars_db.add_row("utterance", u.to_dict())
        return u

    def delete_utterance(self, utterance_id: int) -> None:
        """
        Delete an utterance from the corpus using the in-memory Polars DB.

        Parameters
        ----------
        utterance_id : int
            The ID of the utterance to delete.
        """

        # Retrieve the current utterance table.
        utterance_df = self.polars_db.get_table("utterance")
        # Filter out the utterance with the specified id.
        new_df = utterance_df.filter(pl.col("id") != utterance_id)
        # Replace the utterance table with the updated DataFrame.
        self.polars_db.replace_table("utterance", new_df)

    # def speakers(self, session: Session = None) -> sqlalchemy.orm.Query:
    #     """
    #     Get all speakers in the corpus

    #     Parameters
    #     ----------
    #     session: sqlalchemy.orm.Session, optional
    #        Session to use in querying

    #     Returns
    #     -------
    #     sqlalchemy.orm.Query
    #         Speaker query
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True
    #     speakers = session.query(Speaker).options(
    #         selectinload(Speaker.utterances),
    #         selectinload(Speaker.files),
    #         joinedload(Speaker.dictionary),
    #     )
    #     if close:
    #         session.close()
    #     return speakers

    # def files(self, session: Session = None) -> sqlalchemy.orm.Query:
    #     """
    #     Get all files in the corpus

    #     Parameters
    #     ----------
    #     session: sqlalchemy.orm.Session, optional
    #        Session to use in querying

    #     Returns
    #     -------
    #     sqlalchemy.orm.Query
    #         File query
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True
    #     files = session.query(File).options(
    #         selectinload(File.utterances),
    #         selectinload(File.speakers),
    #         joinedload(File.sound_file),
    #         joinedload(File.text_file),
    #     )
    #     if close:
    #         session.close()
    #     return files

    # def utterances(self, session: Session = None) -> sqlalchemy.orm.Query:
    #     """
    #     Get all utterances in the corpus

    #     Parameters
    #     ----------
    #     session: sqlalchemy.orm.Session, optional
    #        Session to use in querying

    #     Returns
    #     -------
    #     :class:`sqlalchemy.orm.Query`
    #         Utterance query
    #     """
    #     close = False
    #     if session is None:
    #         session = Session(self.db_engine)
    #         close = True
    #     utterances = session.query(Utterance).options(
    #         joinedload(Utterance.file, innerjoin=True),
    #         joinedload(Utterance.speaker, innerjoin=True),
    #         selectinload(Utterance.phone_intervals),
    #         selectinload(Utterance.word_intervals),
    #     )
    #     if close:
    #         session.close()
    #     return utterances
    def speakers(self) -> list[Speaker]:
        """
        Get all speakers in the corpus from the in-memory Polars DB.

        Returns
        -------
        list[Speaker]
            A list of Speaker objects.
        """
        speaker_df = self.polars_db.get_table("speaker")
        # Convert DataFrame rows into Speaker instances.
        return [Speaker.from_dict(record) for record in speaker_df.to_dicts()] if not speaker_df.is_empty() else []

    def files(self) -> list[File]:
        """
        Get all files in the corpus from the in-memory Polars DB.

        Returns
        -------
        list[File]
            A list of File objects.
        """
        file_df = self.polars_db.get_table("file")
        return [File.from_dict(record) for record in file_df.to_dicts()] if not file_df.is_empty() else []

    def utterances(self) -> list[Utterance]:
        """
        Get all utterances in the corpus from the in-memory Polars DB.

        Returns
        -------
        list[Utterance]
            A list of Utterance objects.
        """
        utterance_df = self.polars_db.get_table("utterance")
        return [Utterance.from_dict(record) for record in utterance_df.to_dicts()] if not utterance_df.is_empty() else []

    # def initialize_jobs(self) -> None:
    #     """
    #     Initialize the corpus's Jobs
    #     """

    #     with self.session() as session:
    #         if session.query(sqlalchemy.sql.exists().where(Utterance.job_id > 1)).scalar():
    #             logger.info("Jobs already initialized.")
    #             return
    #         logger.info("Initializing multiprocessing jobs...")
    #         if self.num_speakers < config.NUM_JOBS and not config.SINGLE_SPEAKER:
    #             logger.warning(
    #                 f"Number of jobs was specified as {config.NUM_JOBS}, "
    #                 f"but due to only having {self.num_speakers} speakers, MFA "
    #                 f"will only use {self.num_speakers} jobs. Use the --single_speaker flag if you would like to split "
    #                 f"utterances across jobs regardless of their speaker."
    #             )
    #             config.NUM_JOBS = self.num_speakers
    #             session.query(Job).filter(Job.id > config.NUM_JOBS).delete()
    #             session.query(Corpus).update({Corpus.num_jobs: config.NUM_JOBS})
    #             session.commit()
    #         elif config.SINGLE_SPEAKER and self.num_utterances < config.NUM_JOBS:
    #             logger.warning(
    #                 f"Number of jobs was specified as {config.NUM_JOBS}, "
    #                 f"but due to only having {self.num_utterances} utterances, MFA "
    #                 f"will only use {self.num_utterances} jobs."
    #             )
    #             config.NUM_JOBS = self.num_utterances
    #             session.query(Job).filter(Job.id > config.NUM_JOBS).delete()
    #             session.query(Corpus).update({Corpus.num_jobs: config.NUM_JOBS})
    #             session.commit()

    #         jobs = session.query(Job).all()
    #         update_mappings = []
    #         if config.SINGLE_SPEAKER:
    #             utts_per_job = int(self.num_utterances / config.NUM_JOBS)
    #             if utts_per_job == 0:
    #                 utts_per_job = 1
    #             for i, j in enumerate(jobs):
    #                 update_mappings.extend(
    #                     {"id": u, "job_id": j.id}
    #                     for u in range((utts_per_job * i) + 1, (utts_per_job * (i + 1)) + 1)
    #                 )
    #             last_ind = update_mappings[-1]["id"] + 1
    #             for u in range(last_ind, self.num_utterances):
    #                 update_mappings.append({"id": u, "job_id": jobs[-1].id})
    #             bulk_update(session, Utterance, update_mappings)
    #         else:
    #             utt_counts = {j.id: 0 for j in jobs}
    #             speakers = (
    #                 session.query(Speaker.id, sqlalchemy.func.count(Utterance.id))
    #                 .outerjoin(Speaker.utterances)
    #                 .group_by(Speaker.id)
    #                 .order_by(sqlalchemy.func.count(Utterance.id).desc())
    #             )
    #             for s_id, speaker_utt_count in speakers:
    #                 if not speaker_utt_count:
    #                     continue
    #                 job_id = min(utt_counts.keys(), key=lambda x: utt_counts[x])
    #                 update_mappings.append({"speaker_id": s_id, "job_id": job_id})
    #                 utt_counts[job_id] += speaker_utt_count
    #             bulk_update(session, Utterance, update_mappings, id_field="speaker_id")
    #         session.commit()
    #         if session.query(Dictionary2Job).count() == 0:
    #             dict_job_mappings = []
    #             for job_id, dict_id in (
    #                 session.query(Utterance.job_id, Dictionary.id)
    #                 .join(Utterance.speaker)
    #                 .join(Speaker.dictionary)
    #                 .distinct()
    #             ):
    #                 if not dict_id:
    #                     continue
    #                 dict_job_mappings.append({"job_id": job_id, "dictionary_id": dict_id})
    #             if dict_job_mappings:
    #                 session.execute(Dictionary2Job.insert().values(dict_job_mappings))
    #             session.commit()
    def initialize_jobs(self) -> None:
        """
        Initialize the corpus's Jobs using the in-memory Polars DB.
        """

        # Check if any utterance already has a job_id > 1.
        utterance_df = self.polars_db.get_table("utterance")
        if not utterance_df.filter(pl.col("job_id") > 1).is_empty():
            logger.info("Jobs already initialized.")
            return

        logger.info("Initializing multiprocessing jobs...")

        # Adjust job settings based on number of speakers/utterances and configuration.
        if self.num_speakers < config.NUM_JOBS and not config.SINGLE_SPEAKER:
            logger.warning(
                f"Number of jobs was specified as {config.NUM_JOBS}, "
                f"but due to only having {self.num_speakers} speakers, MFA "
                f"will only use {self.num_speakers} jobs. Use the --single_speaker flag if you would like to split "
                f"utterances across jobs regardless of their speaker."
            )
            config.NUM_JOBS = self.num_speakers

            # Delete any job with id greater than config.NUM_JOBS
            job_df = self.polars_db.get_table("job")
            job_df = job_df.filter(pl.col("id") <= config.NUM_JOBS)
            self.polars_db.replace_table("job", job_df)

            # Update corpus with the new number of jobs
            corpus_df = self.polars_db.get_table("corpus")
            if not corpus_df.is_empty():
                corpus_df = corpus_df.with_columns(pl.lit(config.NUM_JOBS).alias("num_jobs"))
                self.polars_db.replace_table("corpus", corpus_df)

        elif config.SINGLE_SPEAKER and self.num_utterances < config.NUM_JOBS:
            logger.warning(
                f"Number of jobs was specified as {config.NUM_JOBS}, "
                f"but due to only having {self.num_utterances} utterances, MFA "
                f"will only use {self.num_utterances} jobs."
            )
            config.NUM_JOBS = self.num_utterances

            job_df = self.polars_db.get_table("job")
            job_df = job_df.filter(pl.col("id") <= config.NUM_JOBS)
            self.polars_db.replace_table("job", job_df)

            corpus_df = self.polars_db.get_table("corpus")
            if not corpus_df.is_empty():
                corpus_df = corpus_df.with_columns(pl.lit(config.NUM_JOBS).alias("num_jobs"))
                self.polars_db.replace_table("corpus", corpus_df)

        # Retrieve the existing jobs from the Polars DB.
        job_df = self.polars_db.get_table("job")
        jobs = [Job.from_dict(record) for record in job_df.to_dicts()] if not job_df.is_empty() else []

        update_mappings = []

        if config.SINGLE_SPEAKER:
            utts_per_job = int(self.num_utterances / config.NUM_JOBS)
            if utts_per_job == 0:
                utts_per_job = 1
            # For each job, assign utterances in a continuous range.
            for i, j in enumerate(jobs):
                update_mappings.extend(
                    {"id": u, "job_id": j.id}
                    for u in range((utts_per_job * i) + 1, (utts_per_job * (i + 1)) + 1)
                )
            last_ind = update_mappings[-1]["id"] + 1
            for u in range(last_ind, self.num_utterances + 1):
                update_mappings.append({"id": u, "job_id": jobs[-1].id})
            # Update utterances (using 'id' as the key).
            self.polars_db.bulk_update("utterance", update_mappings)
        else:
            # For multi-speaker mode, count utterances per speaker and assign speakers to jobs.
            utt_counts = {j.id: 0 for j in jobs}

            # Group utterances by speaker_id to count utterances.
            utt_counts_df = utterance_df.group_by("speaker_id").agg(
                pl.count("id").alias("utt_count")
            )
            speakers_agg = utt_counts_df.sort("utt_count", descending=True)

            for record in speakers_agg.to_dicts():
                s_id = record["speaker_id"]
                speaker_utt_count = record["utt_count"]
                if not speaker_utt_count:
                    continue
                # Choose the job with the minimum total utterance count.
                job_id = min(utt_counts.keys(), key=lambda key: utt_counts[key])
                update_mappings.append({"speaker_id": s_id, "job_id": job_id})
                utt_counts[job_id] += speaker_utt_count

            # Update utterances using 'speaker_id' as the key.
            self.polars_db.bulk_update("utterance", update_mappings, id_field="speaker_id")

        # (No commit is necessary; changes are directly applied.)

        # Now initialize the dictionary-job mappings if none exist.
        dict_job_df = self.polars_db.get_table("dictionary_job")
        if dict_job_df.height == 0:
            dict_job_mappings = []
            # Join utterance -> speaker -> dictionary.
            speakers_df = self.polars_db.get_table("speaker")
            dictionaries_df = self.polars_db.get_table("dictionary")
            join_df = utterance_df.join(speakers_df, left_on="speaker_id", right_on="id", how="inner")
            join2 = join_df.join(dictionaries_df, left_on="dictionary_id", right_on="id", how="inner")
            # Select distinct job_id and dictionary_id pairs.
            pairs_df = join2.select(["job_id", "dictionary_id"]).unique()
            for row in pairs_df.to_dicts():
                if not row.get("dictionary_id"):
                    continue
                dict_job_mappings.append({"job_id": row["job_id"], "dictionary_id": row["dictionary_id"]})
            if dict_job_mappings:
                new_dict_job_df = pl.DataFrame(dict_job_mappings)
                self.polars_db.replace_table("dictionary_job", new_dict_job_df)

    # def _finalize_load(self, session: Session, import_data: DatabaseImportData):
    #     """Finalize the import of database objects after parsing"""
    #     with session.begin_nested():
    #         c = session.query(Corpus).first()
    #         job_objs = [{"id": j, "corpus_id": c.id} for j in range(1, config.NUM_JOBS + 1)]
    #         session.execute(sqlalchemy.insert(Job.__table__), job_objs)
    #         c.num_jobs = config.NUM_JOBS
    #         if import_data.speaker_objects:
    #             session.execute(sqlalchemy.insert(Speaker.__table__), import_data.speaker_objects)
    #         if import_data.file_objects:
    #             session.execute(sqlalchemy.insert(File.__table__), import_data.file_objects)
    #         if import_data.text_file_objects:
    #             session.execute(
    #                 sqlalchemy.insert(TextFile.__table__), import_data.text_file_objects
    #             )
    #         if import_data.sound_file_objects:
    #             session.execute(
    #                 sqlalchemy.insert(SoundFile.__table__), import_data.sound_file_objects
    #             )
    #         if import_data.speaker_ordering_objects:
    #             session.execute(
    #                 sqlalchemy.insert(SpeakerOrdering),
    #                 import_data.speaker_ordering_objects,
    #             )
    #         if import_data.utterance_objects:
    #             session.execute(
    #                 sqlalchemy.insert(Utterance.__table__), import_data.utterance_objects
    #             )
    #         session.flush()

    #     self.imported = True
    #     speakers = (
    #         session.query(Speaker.id)
    #         .outerjoin(Speaker.utterances)
    #         .group_by(Speaker.id)
    #         .having(sqlalchemy.func.count(Utterance.id) == 0)
    #     )
    #     self._speaker_ids = {}
    #     speaker_ids = [x[0] for x in speakers]
    #     session.query(Corpus).update(
    #         {
    #             "imported": True,
    #             "has_text_files": len(import_data.text_file_objects) > 0,
    #             "has_sound_files": len(import_data.sound_file_objects) > 0,
    #         }
    #     )
    #     if speaker_ids:
    #         session.query(SpeakerOrdering).filter(
    #             SpeakerOrdering.c.speaker_id.in_(speaker_ids)
    #         ).delete()
    #         session.query(Speaker).filter(Speaker.id.in_(speaker_ids)).delete()
    #         self._num_speakers = None
    #     self._num_utterances = None  # Recalculate if already cached
    #     self._num_files = None
    #     session.commit()
    def _finalize_load(self, import_data: DatabaseImportData) -> None:
        """Finalize the import of database objects after parsing using the in-memory Polars DB."""

        # Retrieve the Corpus record (assumes a single row in the corpus table)
        corpus_df = self.polars_db.get_table("corpus")
        if corpus_df.is_empty():
            raise Exception("No corpus found to finalize load.")
        c_dict = corpus_df.to_dicts()[0]
        c = Corpus.from_dict(c_dict)

        # Create job objects and store them in the "job" table
        job_objs = [{"id": j, "corpus_id": c.id} for j in range(1, config.NUM_JOBS + 1)]
        job_df = pl.DataFrame(job_objs)
        self.polars_db.replace_table("job", job_df)

        # Update the corpus record with the number of jobs
        c.num_jobs = config.NUM_JOBS
        updated_corpus = {**c.to_dict(), "num_jobs": config.NUM_JOBS}
        self.polars_db.replace_table("corpus", pl.DataFrame([updated_corpus]))

        # Insert parsed objects into their respective tables if they exist
        if import_data.speaker_objects:
            self.polars_db.replace_table("speaker", pl.DataFrame(import_data.speaker_objects))
        if import_data.file_objects:
            self.polars_db.replace_table("file", pl.DataFrame(import_data.file_objects))
        if import_data.text_file_objects:
            self.polars_db.replace_table("text_file", pl.DataFrame(import_data.text_file_objects))
        if import_data.sound_file_objects:
            self.polars_db.replace_table("sound_file", pl.DataFrame(import_data.sound_file_objects))
        if import_data.speaker_ordering_objects:
            self.polars_db.replace_table("speaker_ordering", pl.DataFrame(import_data.speaker_ordering_objects))
        if import_data.utterance_objects:
            self.polars_db.replace_table("utterance", pl.DataFrame(import_data.utterance_objects))
        # (No explicit flush is needed in an in-memory DB)

        self.imported = True

        # Identify speakers with no utterances:
        speaker_df = self.polars_db.get_table("speaker")
        utterance_df = self.polars_db.get_table("utterance")
        if not speaker_df.is_empty():
            # Count utterances per speaker
            utt_counts = utterance_df.group_by("speaker_id").agg(pl.count("id").alias("utt_count"))
            # Left join speakers with their utterance counts
            merged = speaker_df.join(utt_counts, left_on="id", right_on="speaker_id", how="left")
            merged = merged.with_columns(pl.col("utt_count").fill_null(0))
            # Select speakers where the count is zero
            speakers_no_utt = merged.filter(pl.col("utt_count") == 0)
            speaker_ids = speakers_no_utt.select("id").to_series().to_list() if not speakers_no_utt.is_empty() else []
        else:
            speaker_ids = []

        self._speaker_ids = {}  # Reset speaker ID mapping

        # Update the corpus with import flags and file existence
        corpus_updated = updated_corpus.copy()
        corpus_updated.update({
            "imported": True,
            "has_text_files": len(import_data.text_file_objects) > 0,
            "has_sound_files": len(import_data.sound_file_objects) > 0,
        })
        self.polars_db.replace_table("corpus", pl.DataFrame([corpus_updated]))

        # Remove speaker ordering and speakers for speakers with no utterances
        if speaker_ids:
            spkr_ord_df = self.polars_db.get_table("speaker_ordering")
            if not spkr_ord_df.is_empty():
                filtered_so = spkr_ord_df.filter(~pl.col("speaker_id").is_in(speaker_ids))
                self.polars_db.replace_table("speaker_ordering", filtered_so)
            remaining_speakers = speaker_df.filter(~pl.col("id").is_in(speaker_ids))
            self.polars_db.replace_table("speaker", remaining_speakers)
            self._num_speakers = None

        # Invalidate cached counts
        self._num_utterances = None
        self._num_files = None
        # With an in-memory DB, changes are immediate so no commit is needed.
        
    def get_tokenizers(self):
        from montreal_forced_aligner.dictionary.mixins import DictionaryMixin

        if self.language is Language.unknown:
            tokenizers = getattr(self, "tokenizers", None)
        else:
            from montreal_forced_aligner.tokenization.spacy import (
                check_language_tokenizer_availability,
            )

            check_language_tokenizer_availability(self.language)
            tokenizers = self.language
        if tokenizers is None:
            if isinstance(self, DictionaryMixin):
                tokenizers = self.tokenizer
            else:
                return None
        return tokenizers

    def get_tokenizer(self, dictionary_id: int):
        tokenizers = self.get_tokenizers()
        if not isinstance(tokenizers, dict):
            return tokenizers
        return tokenizers[dictionary_id]

    # def normalize_text_arguments(self):
    #     tokenizers = self.get_tokenizers()
    #     from montreal_forced_aligner.corpus.multiprocessing import NormalizeTextArguments

    #     with self.session() as session:
    #         jobs = session.query(Job).filter(Job.utterances.any())
    #         return [
    #             NormalizeTextArguments(
    #                 j.id,
    #                 getattr(self, "session" if config.USE_THREADING else "db_string", ""),
    #                 self.split_directory.joinpath("log", f"normalize.{j.id}.log"),
    #                 tokenizers,
    #                 getattr(self, "g2p_model", None),
    #                 getattr(self, "ignore_case", True),
    #                 getattr(self, "use_cutoff_model", False),
    #             )
    #             for j in jobs
    #         ]
    def normalize_text_arguments(self):
        tokenizers = self.get_tokenizers()
        from montreal_forced_aligner.corpus.multiprocessing import NormalizeTextArguments

        # Retrieve all jobs from the in-memory Polars DB
        job_df = self.polars_db.get_table("job")
        job_dicts = job_df.to_dicts() if not job_df.is_empty() else []
        # Filter jobs to include only those that have at least one utterance.
        filtered_jobs = [
            Job.from_dict(j) for j in job_dicts if j.get("utterances") and len(j.get("utterances")) > 0
        ]

        return [
            NormalizeTextArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.split_directory.joinpath("log", f"normalize.{j.id}.log"),
                tokenizers,
                getattr(self, "g2p_model", None),
                getattr(self, "ignore_case", True),
                getattr(self, "use_cutoff_model", False),
            )
            for j in filtered_jobs
        ]

    # def normalize_text(self) -> None:
    #     """Normalize the text of the corpus using a dictionary's sanitization functions and word mappings"""
    #     if self.text_normalized:
    #         logger.info("Text already normalized.")
    #         return
    #     args = self.normalize_text_arguments()
    #     if args is None:
    #         return
    #     from montreal_forced_aligner.models import G2PModel

    #     logger.info("Normalizing text...")
    #     log_directory = self.split_directory.joinpath("log")
    #     word_update_mappings = {}
    #     word_insert_mappings = {}
    #     pronunciation_insert_mappings = []
    #     word_indexes = {}
    #     word_mapping_ids = {}
    #     max_mapping_id = 0
    #     log_directory.mkdir(parents=True, exist_ok=True)
    #     update_mapping = []
    #     word_key = self.get_next_primary_key(Word)
    #     g2p_model: G2PModel = getattr(self, "g2p_model", None)
    #     from montreal_forced_aligner.g2p.generator import G2PTopLevelMixin

    #     if isinstance(self, G2PTopLevelMixin):  # G2P happens later
    #         g2p_model = None
    #     pronunciation_key = self.get_next_primary_key(Pronunciation)
    #     with mfa_open(
    #         log_directory.joinpath("normalize_oov.log"), "w"
    #     ) as log_file, self.session() as session:
    #         dictionaries: typing.Dict[int, Dictionary] = {
    #             d.id: d for d in session.query(Dictionary)
    #         }
    #         dict_name_to_id = {v.name: k for k, v in dictionaries.items()}
    #         has_words = (
    #             session.query(Dictionary).filter(Dictionary.name == "unknown").first() is None
    #         )
    #         existing_oovs = {}
    #         words = session.query(
    #             Word.id, Word.mapping_id, Word.dictionary_id, Word.word, Word.word_type
    #         ).order_by(Word.mapping_id)
    #         if not has_words or getattr(self, "use_g2p", False):
    #             word_insert_mappings[(1, "<eps>")] = {
    #                 "id": word_key,
    #                 "word": "<eps>",
    #                 "word_type": WordType.silence,
    #                 "mapping_id": word_key - 1,
    #                 "count": 0,
    #                 "dictionary_id": 1,
    #             }
    #             word_key += 1
    #             max_mapping_id = word_key - 1
    #         for w_id, m_id, d_id, w, wt in words:
    #             if wt is WordType.oov and w not in self.specials_set:
    #                 existing_oovs[(d_id, w)] = {"id": w_id, "count": 0, "included": False}
    #                 continue
    #             word_indexes[(d_id, w)] = w_id
    #             word_mapping_ids[w] = m_id
    #             if m_id > max_mapping_id:
    #                 max_mapping_id = m_id
    #         to_g2p = set()
    #         word_to_g2p_mapping = {x: collections.defaultdict(set) for x in dictionaries.keys()}
    #         word_counts = collections.defaultdict(int)
    #         for result in run_kaldi_function(
    #             NormalizeTextFunction, args, total_count=self.num_utterances
    #         ):
    #             try:
    #                 result, dict_id = result
    #                 if dict_id is None:
    #                     dict_id = list(dictionaries.keys())[0]
    #                 if has_words and not getattr(self, "use_g2p", False) and g2p_model is not None:
    #                     oovs = set(result["oovs"].split())
    #                     pronunciation_text = result["normalized_character_text"].split()
    #                     for i, w in enumerate(result["normalized_text"].split()):
    #                         if (dict_id, w) not in word_indexes:
    #                             if w in dictionaries[dict_id].special_set:
    #                                 continue
    #                             word_counts[(dict_id, w)] += 1
    #                             oovs.add(w)
    #                             if self.language is Language.unknown:
    #                                 to_g2p.add((w, dict_id))
    #                                 word_to_g2p_mapping[dict_id][w].add(w)
    #                             else:
    #                                 to_g2p.add((pronunciation_text[i], dict_id))
    #                                 word_to_g2p_mapping[dict_id][w].add(pronunciation_text[i])
    #                         elif (dict_id, w) not in word_update_mappings:
    #                             word_update_mappings[(dict_id, w)] = {
    #                                 "id": word_indexes[(dict_id, w)],
    #                                 "count": 1,
    #                             }
    #                         else:
    #                             word_update_mappings[(dict_id, w)]["count"] += 1
    #                     result["oovs"] = " ".join(sorted(oovs))
    #                 else:
    #                     for w in result["normalized_text"].split():
    #                         if (dict_id, w) in existing_oovs:
    #                             existing_oovs[(dict_id, w)]["count"] += 1
    #                         elif (dict_id, w) not in word_indexes:
    #                             if (dict_id, w) not in word_insert_mappings:
    #                                 word_insert_mappings[(dict_id, w)] = {
    #                                     "id": word_key,
    #                                     "word": w,
    #                                     "word_type": WordType.oov,
    #                                     "mapping_id": word_key - 1,
    #                                     "count": 0,
    #                                     "dictionary_id": dict_id,
    #                                     "included": False,
    #                                 }
    #                                 pronunciation_insert_mappings.append(
    #                                     {
    #                                         "id": pronunciation_key,
    #                                         "word_id": word_key,
    #                                         "pronunciation": getattr(self, "oov_phone", "spn"),
    #                                     }
    #                                 )
    #                                 word_key += 1
    #                                 pronunciation_key += 1
    #                             word_insert_mappings[(dict_id, w)]["count"] += 1
    #                         elif (dict_id, w) not in word_update_mappings:
    #                             word_update_mappings[(dict_id, w)] = {
    #                                 "id": word_indexes[(dict_id, w)],
    #                                 "count": 1,
    #                             }
    #                         else:
    #                             word_update_mappings[(dict_id, w)]["count"] += 1

    #                 update_mapping.append(result)
    #             except Exception:
    #                 import sys
    #                 import traceback

    #                 exc_type, exc_value, exc_traceback = sys.exc_info()
    #                 logger.debug(
    #                     "\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    #                 )
    #                 raise

    #         bulk_update(session, Utterance, update_mapping)
    #         session.commit()
    #         if word_update_mappings:
    #             if has_words:
    #                 session.query(Word).update({"count": 0})
    #                 session.commit()
    #                 bulk_update(session, Word, list(word_update_mappings.values()))
    #                 session.commit()
    #         with self.session() as session:
    #             if to_g2p:
    #                 log_file.write(f"Found {len(to_g2p)} OOVs\n")
    #                 if g2p_model is not None:
    #                     from montreal_forced_aligner.g2p.generator import PyniniGenerator

    #                     g2pped = {}
    #                     if isinstance(g2p_model, dict):
    #                         for dict_name, g2p_model in g2p_model.items():
    #                             dict_id = dict_name_to_id[dict_name]
    #                             gen = PyniniGenerator(
    #                                 g2p_model_path=g2p_model.source,
    #                                 word_list=[x[0] for x in to_g2p if x[1] == dict_id],
    #                                 num_pronunciations=1,
    #                                 strict_graphemes=True,
    #                             )
    #                             g2pped[dict_id] = gen.generate_pronunciations()
    #                     else:
    #                         gen = PyniniGenerator(
    #                             g2p_model_path=g2p_model.source,
    #                             word_list=[x[0] for x in to_g2p],
    #                             num_pronunciations=1,
    #                             strict_graphemes=True,
    #                         )
    #                         dict_id = list(dictionaries.keys())[0]
    #                         g2pped[dict_id] = gen.generate_pronunciations()
    #                     for dict_id, mapping in word_to_g2p_mapping.items():
    #                         log_file.write(f"For dictionary {dict_id}:\n")
    #                         for w, ps in mapping.items():
    #                             log_file.write(f"  - {w} ({', '.join(sorted(ps))})\n")
    #                             max_mapping_id += 1
    #                             included = False
    #                             if hasattr(self, "brackets") and any(
    #                                 w.startswith(b) for b, _ in self.brackets
    #                             ):
    #                                 word_type = WordType.bracketed
    #                                 pronunciations = [getattr(self, "oov_phone", "spn")]
    #                             else:
    #                                 word_type = WordType.speech
    #                                 if isinstance(g2pped, dict):
    #                                     pronunciations = [
    #                                         g2pped[dict_id][x][0]
    #                                         for x in ps
    #                                         if x in g2pped[dict_id] and g2pped[dict_id][x]
    #                                     ]
    #                                 else:
    #                                     pronunciations = [
    #                                         g2pped[x][0] for x in ps if x in g2pped and g2pped[x]
    #                                     ]
    #                                 if not pronunciations:
    #                                     word_type = WordType.oov
    #                                     pronunciations = [getattr(self, "oov_phone", "spn")]
    #                                 else:
    #                                     included = True

    #                             word_insert_mappings[(dict_id, w)] = {
    #                                 "id": word_key,
    #                                 "mapping_id": max_mapping_id,
    #                                 "word": w,
    #                                 "count": word_counts[(dict_id, w)],
    #                                 "dictionary_id": dict_id,
    #                                 "word_type": word_type,
    #                                 "included": included,
    #                             }
    #                             for p in pronunciations:
    #                                 log_file.write(f"    - {p}\n")
    #                                 pronunciation_insert_mappings.append(
    #                                     {
    #                                         "id": pronunciation_key,
    #                                         "word_id": word_key,
    #                                         "pronunciation": p,
    #                                     }
    #                                 )
    #                                 pronunciation_key += 1
    #                             word_key += 1
    #                 else:
    #                     for word, dict_id in to_g2p:
    #                         if (dict_id, word) in existing_oovs:
    #                             existing_oovs[(dict_id, word)]["count"] += 1
    #                             continue
    #                         if (dict_id, word) not in word_insert_mappings:
    #                             word_insert_mappings[(dict_id, word)] = {
    #                                 "id": word_key,
    #                                 "word": word,
    #                                 "word_type": WordType.oov,
    #                                 "mapping_id": word_key - 1,
    #                                 "count": 0,
    #                                 "included": False,
    #                                 "dictionary_id": dict_id,
    #                             }
    #                             pronunciation_insert_mappings.append(
    #                                 {
    #                                     "id": pronunciation_key,
    #                                     "word_id": word_key,
    #                                     "pronunciation": getattr(self, "oov_phone", "spn"),
    #                                 }
    #                             )
    #                             word_key += 1
    #                             pronunciation_key += 1
    #                         word_insert_mappings[(dict_id, word)]["count"] += 1
    #             log_file.write("Found the following OOVs:\n")
    #             log_file.write(f"{existing_oovs}\n")
    #             log_file.write(f"{word_insert_mappings}\n")
    #             if not has_words:
    #                 word_insert_mappings[(1, "<unk>")] = {
    #                     "id": word_key,
    #                     "word": "<unk>",
    #                     "word_type": WordType.oov,
    #                     "mapping_id": word_key - 1,
    #                     "count": 0,
    #                     "dictionary_id": 1,
    #                 }
    #             if existing_oovs:
    #                 bulk_update(session, Word, list(existing_oovs.values()))
    #                 session.commit()
    #             if word_insert_mappings:
    #                 session.bulk_insert_mappings(
    #                     Word,
    #                     list(word_insert_mappings.values()),
    #                     return_defaults=False,
    #                     render_nulls=True,
    #                 )
    #             if pronunciation_insert_mappings:
    #                 session.bulk_insert_mappings(
    #                     Pronunciation,
    #                     pronunciation_insert_mappings,
    #                     return_defaults=False,
    #                     render_nulls=True,
    #                 )
    #             self.text_normalized = True
    #             session.query(Corpus).update({"text_normalized": True})
    #             session.commit()
    #             if self.oov_count_threshold > 0:
    #                 session.query(Word).filter(Word.word_type == WordType.speech).filter(
    #                     Word.count <= self.oov_count_threshold
    #                 ).update({Word.included: False, Word.word_type: WordType.oov})
    #                 session.commit()
    def normalize_text(self) -> None:
        """Normalize the text of the corpus using a dictionary's sanitization functions and word mappings"""
        if self.text_normalized:
            logger.info("Text already normalized.")
            return

        args = self.normalize_text_arguments()
        if args is None:
            return

        from montreal_forced_aligner.models import G2PModel
        from montreal_forced_aligner.g2p.generator import G2PTopLevelMixin, PyniniGenerator

        logger.info("Normalizing text...")
        log_directory = self.split_directory.joinpath("log")
        log_directory.mkdir(parents=True, exist_ok=True)

        # Initialize mapping variables
        word_update_mappings = {}         # {(dict_id, word): {...}}
        word_insert_mappings = {}         # {(dict_id, word): {...}}
        pronunciation_insert_mappings = []  # [ {...}, ... ]
        word_indexes = {}                 # {(dict_id, word): word_id}
        word_mapping_ids = {}             # {word: mapping_id}
        max_mapping_id = 0
        update_mapping = []               # For updating utterances later

        # Get starting primary keys from the in‑memory DB (pass table name as a string)
        word_key = self.get_next_primary_key("word")
        pronunciation_key = self.get_next_primary_key("pronunciation")

        g2p_model: G2PModel = getattr(self, "g2p_model", None)
        if isinstance(self, G2PTopLevelMixin):
            g2p_model = None

        # Open the log file for OOV (out-of‐vocabulary) words
        with mfa_open(log_directory.joinpath("normalize_oov.log"), "w") as log_file:
            # --- Load the existing dictionaries ---
            dictionaries_df = self.polars_db.get_table("dictionary")
            dictionaries = {d["id"]: Dictionary.from_dict(d) for d in dictionaries_df.to_dicts()}
            dict_name_to_id = {v.name: k for k, v in dictionaries.items()}
            # Determine if a dictionary with name "unknown" exists: if not, then has_words==True
            has_words = dictionaries_df.filter(pl.col("name") == "unknown").is_empty()

            # --- Index current words ---
            existing_oovs = {}
            word_df = self.polars_db.get_table("word")
            # Sort by mapping_id so that updates are applied in order:
            words_df = word_df.sort("mapping_id")
            for row in words_df.to_dicts():
                w_id = row["id"]
                m_id = row["mapping_id"]
                d_id = row["dictionary_id"]
                w = row["word"]
                wt = row["word_type"]
                if wt == WordType.oov and w not in self.specials_set:
                    existing_oovs[(d_id, w)] = {"id": w_id, "count": 0, "included": False}
                    continue
                word_indexes[(d_id, w)] = w_id
                word_mapping_ids[w] = m_id
                if m_id > max_mapping_id:
                    max_mapping_id = m_id

            # --- Prepare G2P structures and word counts ---
            to_g2p = set()
            import collections
            word_to_g2p_mapping = {x: collections.defaultdict(set) for x in dictionaries.keys()}
            word_counts = collections.defaultdict(int)

            # --- Process normalization results ---
            for result_tuple in run_kaldi_function(NormalizeTextFunction, args, total_count=self.num_utterances):
                try:
                    result, dict_id = result_tuple
                    if dict_id is None:
                        dict_id = list(dictionaries.keys())[0]
                    # When using g2p and the dictionary already has words, process normalized texts and OOVs:
                    if has_words and (not getattr(self, "use_g2p", False)) and (g2p_model is not None):
                        oovs = set(result["oovs"].split())
                        pronunciation_text = result["normalized_character_text"].split()
                        normalized_words = result["normalized_text"].split()
                        for i, w in enumerate(normalized_words):
                            if (dict_id, w) not in word_indexes:
                                # Skip special words (if defined in the dictionary)
                                if w in dictionaries[dict_id].special_set:
                                    continue
                                word_counts[(dict_id, w)] += 1
                                oovs.add(w)
                                if self.language is Language.unknown:
                                    to_g2p.add((w, dict_id))
                                    word_to_g2p_mapping[dict_id][w].add(w)
                                else:
                                    to_g2p.add((pronunciation_text[i], dict_id))
                                    word_to_g2p_mapping[dict_id][w].add(pronunciation_text[i])
                            elif (dict_id, w) not in word_update_mappings:
                                word_update_mappings[(dict_id, w)] = {"id": word_indexes[(dict_id, w)], "count": 1}
                            else:
                                word_update_mappings[(dict_id, w)]["count"] += 1
                        result["oovs"] = " ".join(sorted(oovs))
                    else:
                        # Without g2p processing, simply update counts or mark for insertion
                        for w in result["normalized_text"].split():
                            if (dict_id, w) in existing_oovs:
                                existing_oovs[(dict_id, w)]["count"] += 1
                            elif (dict_id, w) not in word_indexes:
                                if (dict_id, w) not in word_insert_mappings:
                                    word_insert_mappings[(dict_id, w)] = {
                                        "id": word_key,
                                        "word": w,
                                        "word_type": WordType.oov,
                                        "mapping_id": word_key - 1,
                                        "count": 0,
                                        "dictionary_id": dict_id,
                                        "included": False,
                                    }
                                    pronunciation_insert_mappings.append({
                                        "id": pronunciation_key,
                                        "word_id": word_key,
                                        "pronunciation": getattr(self, "oov_phone", "spn"),
                                    })
                                    word_key += 1
                                    pronunciation_key += 1
                                word_insert_mappings[(dict_id, w)]["count"] += 1
                            elif (dict_id, w) not in word_update_mappings:
                                word_update_mappings[(dict_id, w)] = {"id": word_indexes[(dict_id, w)], "count": 1}
                            else:
                                word_update_mappings[(dict_id, w)]["count"] += 1

                    update_mapping.append(result)
                except Exception:
                    import sys, traceback
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    logger.debug("\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
                    raise

            # --- Update utterances in the DB (bulk update) ---
            self.polars_db.bulk_update("utterance", update_mapping)

            # --- Update word counts ---
            if word_update_mappings:
                if has_words:
                    # Reset all word counts quickly via vectorized operation:
                    word_df = self.polars_db.get_table("word")
                    if not word_df.is_empty():
                        word_df = word_df.with_columns(pl.lit(0).alias("count"))
                        self.polars_db.replace_table("word", word_df)
                self.polars_db.bulk_update("word", list(word_update_mappings.values()))

            # --- Process G2P OOV words if any ---
            if to_g2p:
                log_file.write(f"Found {len(to_g2p)} OOVs\n")
                if g2p_model is not None:
                    g2pped = {}
                    # If g2p_model is a dict mapping dictionary names to models:
                    if isinstance(g2p_model, dict):
                        for dict_name, model in g2p_model.items():
                            dict_id = dict_name_to_id[dict_name]
                            gen = PyniniGenerator(
                                g2p_model_path=model.source,
                                word_list=[w for w, d in to_g2p if d == dict_id],
                                num_pronunciations=1,
                                strict_graphemes=True,
                            )
                            g2pped[dict_id] = gen.generate_pronunciations()
                    else:
                        gen = PyniniGenerator(
                            g2p_model_path=g2p_model.source,
                            word_list=[w for w, d in to_g2p],
                            num_pronunciations=1,
                            strict_graphemes=True,
                        )
                        dict_id = list(dictionaries.keys())[0]
                        g2pped[dict_id] = gen.generate_pronunciations()
                    for dict_id, mapping in word_to_g2p_mapping.items():
                        log_file.write(f"For dictionary {dict_id}:\n")
                        for w, ps in mapping.items():
                            log_file.write(f"  - {w} ({', '.join(sorted(ps))})\n")
                            max_mapping_id += 1
                            included = False
                            if hasattr(self, "brackets") and any(w.startswith(b) for b, _ in self.brackets):
                                word_type = WordType.bracketed
                                pronunciations = [getattr(self, "oov_phone", "spn")]
                            else:
                                word_type = WordType.speech
                                if isinstance(g2pped, dict):
                                    pronunciations = [g2pped[dict_id][x][0] 
                                                    for x in ps 
                                                    if x in g2pped[dict_id] and g2pped[dict_id][x]]
                                else:
                                    pronunciations = [g2pped[x][0] 
                                                    for x in ps 
                                                    if x in g2pped and g2pped[x]]
                                if not pronunciations:
                                    word_type = WordType.oov
                                    pronunciations = [getattr(self, "oov_phone", "spn")]
                                else:
                                    included = True
                            word_insert_mappings[(dict_id, w)] = {
                                "id": word_key,
                                "mapping_id": max_mapping_id,
                                "word": w,
                                "count": word_counts[(dict_id, w)],
                                "dictionary_id": dict_id,
                                "word_type": word_type,
                                "included": included,
                            }
                            for p in pronunciations:
                                log_file.write(f"    - {p}\n")
                                pronunciation_insert_mappings.append({
                                    "id": pronunciation_key,
                                    "word_id": word_key,
                                    "pronunciation": p,
                                })
                                pronunciation_key += 1
                            word_key += 1
                else:
                    # If no g2p model, increment counts for OOV words
                    for word, dict_id in to_g2p:
                        if (dict_id, word) in existing_oovs:
                            existing_oovs[(dict_id, word)]["count"] += 1
                            continue
                        if (dict_id, word) not in word_insert_mappings:
                            word_insert_mappings[(dict_id, word)] = {
                                "id": word_key,
                                "word": word,
                                "word_type": WordType.oov,
                                "mapping_id": word_key - 1,
                                "count": 0,
                                "included": False,
                                "dictionary_id": dict_id,
                            }
                            pronunciation_insert_mappings.append({
                                "id": pronunciation_key,
                                "word_id": word_key,
                                "pronunciation": getattr(self, "oov_phone", "spn"),
                            })
                            word_key += 1
                            pronunciation_key += 1
                        word_insert_mappings[(dict_id, word)]["count"] += 1

                log_file.write("Found the following OOVs:\n")
                log_file.write(f"{existing_oovs}\n")
                log_file.write(f"{word_insert_mappings}\n")
                if not has_words:
                    word_insert_mappings[(1, "<unk>")] = {
                        "id": word_key,
                        "word": "<unk>",
                        "word_type": WordType.oov,
                        "mapping_id": word_key - 1,
                        "count": 0,
                        "dictionary_id": 1,
                    }

                if existing_oovs:
                    self.polars_db.bulk_update("word", list(existing_oovs.values()))
                if word_insert_mappings:
                    self.polars_db.bulk_insert("word", list(word_insert_mappings.values()))
                if pronunciation_insert_mappings:
                    self.polars_db.bulk_insert("pronunciation", pronunciation_insert_mappings)

            # --- Final updates ---
            self.text_normalized = True
            corpus_df = self.polars_db.get_table("corpus")
            if not corpus_df.is_empty():
                corpus_dict = corpus_df.to_dicts()[0]
                corpus_dict["text_normalized"] = True
                self.polars_db.replace_table("corpus", pl.DataFrame([corpus_dict]))
            if getattr(self, "oov_count_threshold", 0) > 0:
                # Optimize by updating in a vectorized way:
                word_df = self.polars_db.get_table("word")
                if not word_df.is_empty():
                    mask = (word_df["word_type"] == WordType.speech) & (word_df["count"] <= self.oov_count_threshold)
                    if mask.sum() > 0:
                        # Set the columns 'included' and 'word_type' for matched rows
                        word_df = word_df.with_columns([
                            pl.when(mask).then(False).otherwise(pl.col("included")).alias("included"),
                            pl.when(mask).then(WordType.oov).otherwise(pl.col("word_type")).alias("word_type")
                        ])
                        self.polars_db.replace_table("word", word_df)
                        
    # def add_speaker(self, name: str, session: Session = None):
    #     """
    #     Add a speaker to the corpus

    #     Parameters
    #     ----------
    #     name: str
    #         Name of the speaker
    #     session: sqlalchemy.orm.Session
    #         Database session, if not specified, will use a temporary session

    #     """
    #     if name in self._speaker_ids:
    #         return
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True

    #     speaker_obj = session.query(Speaker).filter_by(name=name).first()
    #     if not speaker_obj:
    #         dictionary = None
    #         if hasattr(self, "get_dictionary_id"):
    #             dictionary = session.get(Dictionary, self.get_dictionary_id(name))
    #         speaker_obj = Speaker(
    #             id=self.get_next_primary_key(Speaker), name=name, dictionary=dictionary
    #         )
    #         session.add(speaker_obj)
    #         session.flush()
    #         self._speaker_ids[name] = speaker_obj.id
    #     else:
    #         self._speaker_ids[name] = speaker_obj.id

    #     if close:
    #         session.commit()
    #         session.close()

    # def _create_dummy_dictionary(self):
    #     with self.session() as session:
    #         if session.query(Dictionary).first() is None:
    #             dialect = Dialect(name="unspecified")
    #             d = Dictionary(name="unknown", path="unknown", dialect=dialect)
    #             session.add(dialect)
    #             session.add(d)
    #             session.flush()
    #             session.query(Speaker).update({Speaker.dictionary_id: d.id})
    #             session.commit()
    def add_speaker(self, name: str):
        """
        Add a speaker to the corpus using the in-memory Polars DB.

        Parameters
        ----------
        name : str
            Name of the speaker.
        """

        # If the speaker has already been added, no need to proceed.
        if name in self._speaker_ids:
            return

        # Look up the speaker in the in-memory "speaker" table.
        speaker_df = self.polars_db.get_table("speaker").filter(pl.col("name") == name)
        if not speaker_df.is_empty():
            speaker_obj = Speaker.from_dict(speaker_df.to_dicts()[0])
            self._speaker_ids[name] = speaker_obj.id
            return

        # Speaker not found: create a new speaker.
        dictionary = None
        # If the corpus has a method to look up a dictionary ID for a speaker,
        # retrieve the corresponding dictionary from the in-memory DB.
        if hasattr(self, "get_dictionary_id"):
            dict_id = self.get_dictionary_id(name)
            dict_df = self.polars_db.get_table("dictionary").filter(pl.col("id") == dict_id)
            if not dict_df.is_empty():
                dictionary = Dictionary.from_dict(dict_df.to_dicts()[0])

        # Create new speaker, assigning a new primary key.
        new_id = self.get_next_primary_key("speaker")  # or self.get_next_primary_key(Speaker)
        speaker_obj = Speaker(id=new_id, name=name, dictionary=dictionary)
        self.polars_db.add_row("speaker", speaker_obj.to_dict())
        self._speaker_ids[name] = speaker_obj.id


    def _create_dummy_dictionary(self):
        """
        Create a dummy dictionary if none exists and update all speakers to reference it.
        """

        dictionary_df = self.polars_db.get_table("dictionary")
        if dictionary_df.is_empty():
            # Create a dummy dialect and dictionary.
            dialect = Dialect(name="unspecified")
            dummy_dict = Dictionary(name="unknown", path="unknown", dialect=dialect)
            # Add the new dialect and dictionary to the in-memory DB.
            self.polars_db.add_row("dialect", dialect.to_dict())
            self.polars_db.add_row("dictionary", dummy_dict.to_dict())
            # Vectorized update: set the dictionary_id for all speakers.
            speaker_df = self.polars_db.get_table("speaker")
            if not speaker_df.is_empty():
                updated_speaker_df = speaker_df.with_columns(
                    pl.lit(dummy_dict.id).alias("dictionary_id")
                )
                self.polars_db.replace_table("speaker", updated_speaker_df)
                
    # def add_file(self, file: FileData, session: Session = None):
    #     """
    #     Add a file to the corpus

    #     Parameters
    #     ----------
    #     file: :class:`~montreal_forced_aligner.corpus.classes.FileData`
    #         File to be added
    #     """
    #     close = False
    #     if session is None:
    #         session = self.session()
    #         close = True
    #     f = File(
    #         id=self._current_file_index,
    #         name=file.name,
    #         relative_path=file.relative_path,
    #         modified=False,
    #     )
    #     session.add(f)
    #     session.flush()
    #     for i, speaker in enumerate(file.speaker_ordering):
    #         if speaker not in self._speaker_ids:
    #             speaker_obj = Speaker(
    #                 id=self._current_speaker_index,
    #                 name=speaker,
    #                 dictionary_id=getattr(self, "_default_dictionary_id", None),
    #             )
    #             session.add(speaker_obj)
    #             self._speaker_ids[speaker] = self._current_speaker_index
    #             self._current_speaker_index += 1

    #         so = SpeakerOrdering(
    #             file_id=self._current_file_index,
    #             speaker_id=self._speaker_ids[speaker],
    #             index=i,
    #         )
    #         session.add(so)
    #     if file.wav_path is not None:
    #         sf = SoundFile(
    #             file_id=self._current_file_index,
    #             sound_file_path=file.wav_path,
    #             format=file.wav_info.format,
    #             sample_rate=file.wav_info.sample_rate,
    #             duration=file.wav_info.duration,
    #             num_channels=file.wav_info.num_channels,
    #         )
    #         session.add(sf)
    #     if file.text_path is not None:
    #         text_type = file.text_type
    #         if isinstance(text_type, TextFileType):
    #             text_type = file.text_type.value

    #         tf = TextFile(
    #             file_id=self._current_file_index,
    #             text_file_path=file.text_path,
    #             file_type=text_type,
    #         )
    #         session.add(tf)
    #     frame_shift = getattr(self, "frame_shift", None)
    #     if frame_shift is not None:
    #         frame_shift = round(frame_shift / 1000, 4)
    #     for u in file.utterances:
    #         duration = u.end - u.begin
    #         num_frames = None
    #         if frame_shift is not None:
    #             num_frames = int(duration / frame_shift)
    #         utterance = Utterance(
    #             id=self._current_utterance_index,
    #             begin=u.begin,
    #             end=u.end,
    #             duration=duration,
    #             channel=u.channel,
    #             oovs=u.oovs,
    #             normalized_text=u.normalized_text,
    #             normalized_character_text=u.normalized_character_text,
    #             text=u.text,
    #             num_frames=num_frames,
    #             in_subset=False,
    #             ignored=False,
    #             file_id=self._current_file_index,
    #             speaker_id=self._speaker_ids[u.speaker_name],
    #         )
    #         self._current_utterance_index += 1
    #         session.add(utterance)

    #     if close:
    #         session.commit()
    #         session.close()
    #     self._current_file_index += 1
    def add_file(self, file: FileData):
        """
        Add a file to the corpus using the in-memory Polars DB.

        Parameters
        ----------
        file : FileData
            File to be added.
        """
        # Create the file record with the current file index and add it.
        f = File(
            id=self._current_file_index,
            name=file.name,
            relative_path=file.relative_path,
            modified=False,
        )
        self.polars_db.add_row("file", f.to_dict())

        # Process speakers in the file's speaker ordering.
        speaker_ordering_rows = []
        for i, speaker in enumerate(file.speaker_ordering):
            # If the speaker doesn't exist in our mapping, add them.
            if speaker not in self._speaker_ids:
                speaker_obj = Speaker(
                    id=self._current_speaker_index,
                    name=speaker,
                    dictionary_id=getattr(self, "_default_dictionary_id", None),
                )
                self.polars_db.add_row("speaker", speaker_obj.to_dict())
                self._speaker_ids[speaker] = self._current_speaker_index
                self._current_speaker_index += 1

            # Create a speaker ordering record.
            so = SpeakerOrdering(
                file_id=self._current_file_index,
                speaker_id=self._speaker_ids[speaker],
                index=i,
            )
            speaker_ordering_rows.append(so.to_dict())
        
        # Bulk insert speaker ordering records.
        if speaker_ordering_rows:
            self.polars_db.bulk_insert("speaker_ordering", speaker_ordering_rows)

        # Insert a sound file record if a WAV path is specified.
        if file.wav_path is not None:
            sf = SoundFile(
                file_id=self._current_file_index,
                sound_file_path=file.wav_path,
                format=file.wav_info.format,
                sample_rate=file.wav_info.sample_rate,
                duration=file.wav_info.duration,
                num_channels=file.wav_info.num_channels,
            )
            self.polars_db.add_row("sound_file", sf.to_dict())

        # Insert a text file record if a text path is present.
        if file.text_path is not None:
            text_type = file.text_type.value if isinstance(file.text_type, TextFileType) else file.text_type
            tf = TextFile(
                file_id=self._current_file_index,
                text_file_path=file.text_path,
                file_type=text_type,
            )
            self.polars_db.add_row("text_file", tf.to_dict())

        # Compute the frame_shift value and use it to determine the number of frames for each utterance.
        frame_shift = getattr(self, "frame_shift", None)
        if frame_shift is not None:
            frame_shift = round(frame_shift / 1000, 4)

        # Accumulate utterance records to insert them in bulk.
        utterance_rows = []
        for u in file.utterances:
            duration = u.end - u.begin
            num_frames = int(duration / frame_shift) if frame_shift is not None else None

            utterance = Utterance(
                id=self._current_utterance_index,
                begin=u.begin,
                end=u.end,
                duration=duration,
                channel=u.channel,
                oovs=u.oovs,
                normalized_text=u.normalized_text,
                normalized_character_text=u.normalized_character_text,
                text=u.text,
                num_frames=num_frames,
                in_subset=False,
                ignored=False,
                file_id=self._current_file_index,
                speaker_id=self._speaker_ids[u.speaker_name],
            )
            utterance_rows.append(utterance.to_dict())
            self._current_utterance_index += 1

        if utterance_rows:
            self.polars_db.bulk_insert("utterance", utterance_rows)

        # Increment the current file index for the next inserted file.
        self._current_file_index += 1
        
    def generate_import_objects(self, file: FileData) -> DatabaseImportData:
        """
        Add a file to the corpus

        Parameters
        ----------
        file: :class:`~montreal_forced_aligner.corpus.classes.FileData`
            File to be added
        """
        data = DatabaseImportData()
        data.file_objects.append(
            {
                "id": self._current_file_index,
                "name": file.name,
                "relative_path": file.relative_path,
                "modified": False,
            }
        )
        for i, speaker in enumerate(file.speaker_ordering):
            if speaker not in self._speaker_ids:
                data.speaker_objects.append(
                    {
                        "id": self._current_speaker_index,
                        "name": speaker,
                        "dictionary_id": getattr(self, "_default_dictionary_id", None),
                    }
                )
                self._speaker_ids[speaker] = self._current_speaker_index
                self._current_speaker_index += 1

            data.speaker_ordering_objects.append(
                {
                    "file_id": self._current_file_index,
                    "speaker_id": self._speaker_ids[speaker],
                    "index": i,
                }
            )
        if file.wav_path is not None:
            data.sound_file_objects.append(
                {
                    "file_id": self._current_file_index,
                    "sound_file_path": file.wav_path,
                    "format": file.wav_info.format,
                    "sample_rate": file.wav_info.sample_rate,
                    "duration": file.wav_info.duration,
                    "num_channels": file.wav_info.num_channels,
                }
            )
        if file.text_path is not None:
            text_type = file.text_type
            if isinstance(text_type, TextFileType):
                text_type = file.text_type.value

            data.text_file_objects.append(
                {
                    "file_id": self._current_file_index,
                    "text_file_path": file.text_path,
                    "file_type": text_type,
                }
            )
        frame_shift = getattr(self, "frame_shift", None)
        if frame_shift is not None:
            frame_shift = round(frame_shift / 1000, 4)
        for u in file.utterances:
            duration = u.end - u.begin
            num_frames = None
            if frame_shift is not None:
                num_frames = int(duration / frame_shift)
            ignored = False
            if self.ignore_empty_utterances and not u.text:
                ignored = True
            data.utterance_objects.append(
                {
                    "id": self._current_utterance_index,
                    "begin": u.begin,
                    "end": u.end,
                    "channel": u.channel,
                    "oovs": u.oovs,
                    "normalized_text": u.normalized_text,
                    "normalized_character_text": u.normalized_character_text,
                    "text": u.text,
                    "num_frames": num_frames,
                    "in_subset": False,
                    "ignored": ignored,
                    "file_id": self._current_file_index,
                    "job_id": 1,
                    "speaker_id": self._speaker_ids[u.speaker_name],
                }
            )
            self._current_utterance_index += 1
        self._current_file_index += 1
        return data

    @property
    def data_source_identifier(self) -> str:
        """Corpus name"""
        return os.path.basename(self.corpus_directory)

    # def create_subset(self, subset: int) -> None:
    #     """
    #     Create a subset of utterances to use for training

    #     Parameters
    #     ----------
    #     subset: int
    #         Number of utterances to include in subset
    #     """
    #     logger.info(f"Creating subset directory with {subset} utterances...")
    #     if hasattr(self, "cutoff_word") and hasattr(self, "brackets"):
    #         initial_brackets = re.escape("".join(x[0] for x in self.brackets))
    #         final_brackets = re.escape("".join(x[1] for x in self.brackets))
    #         cutoff_identifier = re.sub(
    #             rf"[{initial_brackets}{final_brackets}]", "", self.cutoff_word
    #         )
    #         cutoff_pattern = f"[{initial_brackets}]({cutoff_identifier}|hes)"
    #     else:
    #         cutoff_pattern = "<(cutoff|hes)"

    #     def add_filters(query):
    #         subset_word_count = getattr(self, "subset_word_count", 3)
    #         multiword_pattern = rf"(\s\S+){{{subset_word_count},}}"
    #         filtered = (
    #             query.filter(
    #                 Utterance.normalized_text.op("~")(multiword_pattern)
    #                 if config.USE_POSTGRES
    #                 else Utterance.normalized_text.regexp_match(multiword_pattern)
    #             )
    #             .filter(Utterance.ignored == False)  # noqa
    #             .filter(
    #                 sqlalchemy.or_(
    #                     Utterance.duration_deviation == None,  # noqa
    #                     Utterance.duration_deviation < 10,
    #                 )
    #             )
    #         )
    #         if subset <= 25000:
    #             filtered = filtered.filter(
    #                 sqlalchemy.not_(
    #                     Utterance.normalized_text.op("~")(cutoff_pattern)
    #                     if config.USE_POSTGRES
    #                     else Utterance.normalized_text.regexp_match(cutoff_pattern)
    #                 )
    #             )

    #         return filtered

    #     with self.session() as session:
    #         begin = time.time()
    #         session.query(Utterance).filter(Utterance.in_subset == True).update(  # noqa
    #             {Utterance.in_subset: False}
    #         )
    #         session.commit()
    #         dictionary_query = session.query(Dictionary.name, Dictionary.id).filter(
    #             Dictionary.name != "default"
    #         )
    #         if subset <= 25000:
    #             dictionary_query = dictionary_query.filter(Dictionary.name != "nonnative")
    #         dictionary_lookup = {k: v for k, v in dictionary_query}
    #         num_dictionaries = len(dictionary_lookup)
    #         if num_dictionaries > 1:
    #             subsets_per_dictionary = {}
    #             utts_per_dictionary = {}
    #             subsetted = 0
    #             for dict_name, dict_id in dictionary_lookup.items():
    #                 base_query = (
    #                     session.query(Utterance)
    #                     .join(Utterance.speaker)
    #                     .filter(Speaker.dictionary_id == dict_id)  # noqa
    #                 )
    #                 base_query = add_filters(base_query)
    #                 num_utts = base_query.count()
    #                 utts_per_dictionary[dict_name] = num_utts
    #                 if num_utts < int(subset / num_dictionaries):
    #                     subsets_per_dictionary[dict_name] = num_utts
    #                     subsetted += 1
    #             remaining_subset = subset - sum(subsets_per_dictionary.values())
    #             remaining_dicts = num_dictionaries - subsetted
    #             remaining_subset_per_dictionary = int(remaining_subset / remaining_dicts)
    #             for dict_name, num_utts in sorted(utts_per_dictionary.items(), key=lambda x: x[1]):
    #                 dict_id = dictionary_lookup[dict_name]
    #                 if dict_name in subsets_per_dictionary:
    #                     subset_per_dictionary = subsets_per_dictionary[dict_name]
    #                 else:
    #                     subset_per_dictionary = remaining_subset_per_dictionary
    #                     remaining_dicts -= 1
    #                     if remaining_dicts > 0:
    #                         if num_utts < subset_per_dictionary:
    #                             remaining_subset -= num_utts
    #                         else:
    #                             remaining_subset -= subset_per_dictionary
    #                         remaining_subset_per_dictionary = int(
    #                             remaining_subset / remaining_dicts
    #                         )
    #                 logger.debug(f"For {dict_name}, total number of utterances is {num_utts}")
    #                 larger_subset_num = int(subset_per_dictionary * 10)
    #                 speaker_ids = None
    #                 average_duration = (
    #                     add_filters(
    #                         session.query(sqlalchemy.func.avg(Utterance.duration))
    #                         .join(Utterance.speaker)
    #                         .filter(Speaker.dictionary_id == dict_id)
    #                     )
    #                 ).first()[0]
    #                 for utt_count_cutoff in [30, 15, 5]:
    #                     sq = (
    #                         add_filters(
    #                             session.query(
    #                                 Speaker.id.label("speaker_id"),
    #                                 sqlalchemy.func.count(Utterance.id).label("utt_count"),
    #                             )
    #                             .join(Utterance.speaker)
    #                             .filter(Speaker.dictionary_id == dict_id)
    #                         )
    #                         .filter(Utterance.duration <= average_duration)
    #                         .group_by(Speaker.id.label("speaker_id"))
    #                         .subquery()
    #                     )
    #                     total_speaker_utterances = (
    #                         session.query(sqlalchemy.func.sum(sq.c.utt_count)).filter(
    #                             sq.c.utt_count >= utt_count_cutoff
    #                         )
    #                     ).first()[0]
    #                     if total_speaker_utterances >= subset_per_dictionary:
    #                         speaker_ids = [
    #                             x
    #                             for x, in session.query(sq.c.speaker_id).filter(
    #                                 sq.c.utt_count >= utt_count_cutoff
    #                             )
    #                         ]
    #                         break
    #                 if num_utts > larger_subset_num:
    #                     larger_subset_query = (
    #                         session.query(Utterance.id)
    #                         .join(Utterance.speaker)
    #                         .filter(Speaker.dictionary_id == dict_id)  # noqa
    #                     )
    #                     larger_subset_query = add_filters(larger_subset_query)
    #                     if speaker_ids:
    #                         larger_subset_query = larger_subset_query.filter(
    #                             Speaker.id.in_(speaker_ids)
    #                         )
    #                     larger_subset_query = larger_subset_query.order_by(
    #                         Utterance.duration
    #                     ).limit(larger_subset_num)
    #                     sq = larger_subset_query.subquery()
    #                     subset_utts = (
    #                         sqlalchemy.select(sq.c.id)
    #                         .order_by(sqlalchemy.func.random())
    #                         .limit(subset_per_dictionary)
    #                         .scalar_subquery()
    #                     )
    #                     query = (
    #                         sqlalchemy.update(Utterance)
    #                         .execution_options(synchronize_session="fetch")
    #                         .values(in_subset=True)
    #                         .where(Utterance.id.in_(subset_utts))
    #                     )
    #                     session.execute(query)

    #                     # Remove speakers with less than 5 utterances from subset,
    #                     # can't estimate speaker transforms well for low utterance counts
    #                     sq = (
    #                         session.query(
    #                             Utterance.speaker_id.label("speaker_id"),
    #                             sqlalchemy.func.count(Utterance.id).label("utt_count"),
    #                         )
    #                         .filter(Utterance.in_subset == True)  # noqa
    #                         .group_by(Utterance.speaker_id.label("speaker_id"))
    #                         .subquery()
    #                     )
    #                     speaker_ids = [
    #                         x for x, in session.query(sq.c.speaker_id).filter(sq.c.utt_count < 5)
    #                     ]
    #                     session.query(Utterance).filter(
    #                         Utterance.speaker_id.in_(speaker_ids)
    #                     ).update({Utterance.in_subset: False})
    #                     session.commit()
    #                     logger.debug(f"For {dict_name}, subset is {subset_per_dictionary}")
    #                 elif num_utts > subset_per_dictionary:
    #                     larger_subset_query = (
    #                         session.query(Utterance.id)
    #                         .join(Utterance.speaker)
    #                         .filter(Speaker.dictionary_id == dict_id)  # noqa
    #                     )
    #                     larger_subset_query = add_filters(larger_subset_query)
    #                     if speaker_ids:
    #                         larger_subset_query = larger_subset_query.filter(
    #                             Speaker.id.in_(speaker_ids)
    #                         )
    #                     sq = larger_subset_query.subquery()
    #                     subset_utts = (
    #                         sqlalchemy.select(sq.c.id)
    #                         .order_by(sqlalchemy.func.random())
    #                         .limit(subset_per_dictionary)
    #                         .scalar_subquery()
    #                     )
    #                     query = (
    #                         sqlalchemy.update(Utterance)
    #                         .execution_options(synchronize_session="fetch")
    #                         .values(in_subset=True)
    #                         .where(Utterance.id.in_(subset_utts))
    #                     )
    #                     session.execute(query)
    #                     session.commit()

    #                     logger.debug(f"For {dict_name}, subset is {subset_per_dictionary}")
    #                 else:
    #                     larger_subset_query = (
    #                         session.query(Utterance.id)
    #                         .join(Utterance.speaker)
    #                         .filter(Speaker.dictionary_id == dict_id)
    #                         .filter(Utterance.ignored == False)  # noqa
    #                         .filter(
    #                             sqlalchemy.or_(
    #                                 Utterance.duration_deviation == None,  # noqa
    #                                 Utterance.duration_deviation < 10,
    #                             )
    #                         )  # noqa
    #                     )
    #                     sq = larger_subset_query.subquery()
    #                     subset_utts = sqlalchemy.select(sq.c.id).scalar_subquery()
    #                     query = (
    #                         sqlalchemy.update(Utterance)
    #                         .execution_options(synchronize_session="fetch")
    #                         .values(in_subset=True)
    #                         .where(Utterance.id.in_(subset_utts))
    #                     )
    #                     session.execute(query)
    #                     session.commit()

    #                 # Reassign any utterances from speakers below utterance count threshold
    #                 sq = (
    #                     session.query(
    #                         Utterance.speaker_id.label("speaker_id"),
    #                         sqlalchemy.func.count(Utterance.id).label("utt_count"),
    #                     )
    #                     .join(Utterance.speaker)
    #                     .filter(Speaker.dictionary_id == dict_id)
    #                     .filter(Utterance.in_subset == True)  # noqa
    #                     .group_by(Utterance.speaker_id.label("speaker_id"))
    #                     .subquery()
    #                 )
    #                 total_speaker_utterances = session.query(
    #                     sqlalchemy.func.sum(sq.c.utt_count)
    #                 ).first()[0]
    #                 remaining = subset_per_dictionary - total_speaker_utterances
    #                 if remaining > 0:
    #                     speaker_ids = [x for x, in session.query(sq.c.speaker_id)]

    #                     larger_subset_query = (
    #                         session.query(Utterance.id)
    #                         .join(Utterance.speaker)
    #                         .filter(Speaker.dictionary_id == dict_id)  # noqa
    #                     )
    #                     larger_subset_query = add_filters(larger_subset_query)
    #                     if speaker_ids:
    #                         larger_subset_query = larger_subset_query.filter(
    #                             Speaker.id.in_(speaker_ids)
    #                         )
    #                     larger_subset_query = larger_subset_query.order_by(
    #                         Utterance.duration
    #                     ).limit(remaining * 10)
    #                     sq = larger_subset_query.subquery()
    #                     subset_utts = (
    #                         sqlalchemy.select(sq.c.id)
    #                         .order_by(sqlalchemy.func.random())
    #                         .limit(remaining)
    #                         .scalar_subquery()
    #                     )
    #                     query = (
    #                         sqlalchemy.update(Utterance)
    #                         .execution_options(synchronize_session="fetch")
    #                         .values(in_subset=True)
    #                         .where(Utterance.id.in_(subset_utts))
    #                     )
    #                     session.execute(query)

    #         else:
    #             larger_subset_num = subset * 10
    #             if subset < self.num_utterances:
    #                 # Get all shorter utterances that are not one word long
    #                 larger_subset_query = (
    #                     add_filters(session.query(Utterance.id))
    #                     .order_by(Utterance.duration)
    #                     .limit(larger_subset_num)
    #                 )
    #                 sq = larger_subset_query.subquery()
    #                 subset_utts = (
    #                     sqlalchemy.select(sq.c.id)
    #                     .order_by(sqlalchemy.func.random())
    #                     .limit(subset)
    #                     .scalar_subquery()
    #                 )
    #                 query = (
    #                     sqlalchemy.update(Utterance)
    #                     .execution_options(synchronize_session="fetch")
    #                     .values(in_subset=True)
    #                     .where(Utterance.id.in_(subset_utts))
    #                 )
    #                 session.execute(query)
    #             else:
    #                 session.query(Utterance).update({Utterance.in_subset: True})

    #         session.commit()
    #         subset_directory = self.corpus_output_directory.joinpath(f"subset_{subset}")
    #         log_dir = subset_directory.joinpath("log")
    #         os.makedirs(log_dir, exist_ok=True)

    #         logger.debug(f"Setting subset flags took {time.time() - begin} seconds")
    #         with self.session() as session:
    #             jobs = (
    #                 session.query(Job)
    #                 .options(
    #                     joinedload(Job.corpus, innerjoin=True), subqueryload(Job.dictionaries)
    #                 )
    #                 .filter(Job.utterances.any(Utterance.in_subset == True))  # noqa
    #             )
    #             self._jobs = jobs.all()
    #             arguments = [
    #                 ExportKaldiFilesArguments(
    #                     j.id,
    #                     getattr(self, "session" if config.USE_THREADING else "db_string", ""),
    #                     None,
    #                     subset_directory,
    #                 )
    #                 for j in self._jobs
    #             ]
    #         for _ in run_kaldi_function(ExportKaldiFilesFunction, arguments, total_count=subset):
    #             pass
    def create_subset(self, subset: int) -> None:
        """
        Create a subset of utterances to use for training using the Polars in‑memory DB.
        
        Parameters
        ----------
        subset : int
            Number of utterances to include in the subset.
        """

        logger.info(f"Creating subset directory with {subset} utterances...")

        # Compute cutoff pattern based on corpus attributes (if available)
        if hasattr(self, "cutoff_word") and hasattr(self, "brackets"):
            initial_brackets = re.escape("".join(x[0] for x in self.brackets))
            final_brackets = re.escape("".join(x[1] for x in self.brackets))
            cutoff_identifier = re.sub(rf"[{initial_brackets}{final_brackets}]", "", self.cutoff_word)
            cutoff_pattern = f"[{initial_brackets}]({cutoff_identifier}|hes)"
        else:
            cutoff_pattern = "<(cutoff|hes)"

        # Local helper: given a Polars DataFrame of utterances (with any joined speaker fields),
        # apply the desired filters.
        def add_filters(utt_df: pl.DataFrame) -> pl.DataFrame:
            subset_word_count = getattr(self, "subset_word_count", 3)
            multiword_pattern = rf"(\s\S+){{{subset_word_count},}}"
            # Keep utterances whose normalized text matches the multiword pattern,
            # whose 'ignored' flag is False, and whose duration deviation is either null or small.
            filtered = (
                utt_df.filter(pl.col("normalized_text").str.contains(multiword_pattern))
                    .filter(pl.col("ignored") == False)
                    .filter(pl.col("duration_deviation").is_null() | (pl.col("duration_deviation") < 10))
            )
            if subset <= 25000:
                # Exclude utterances that match the cutoff pattern.
                filtered = filtered.filter(~pl.col("normalized_text").str.contains(cutoff_pattern))
            return filtered

        begin = time.time()

        # Reset the 'in_subset' flag on every utterance
        utt_df = self.polars_db.get_table("utterance")
        utt_df = utt_df.with_columns(pl.lit(False).alias("in_subset"))
        self.polars_db.replace_table("utterance", utt_df)

        # Query dictionaries and build a lookup of (name -> id)
        dict_df = self.polars_db.get_table("dictionary")
        dict_filtered = dict_df.filter(pl.col("name") != "default")
        if subset <= 25000:
            dict_filtered = dict_filtered.filter(pl.col("name") != "nonnative")
        dictionary_lookup = {row["name"]: row["id"] for row in dict_filtered.to_dicts()}
        num_dictionaries = len(dictionary_lookup)

        # Grab current speakers and utterances tables; then join utterances with speakers
        speaker_df = self.polars_db.get_table("speaker")
        utt_df = self.polars_db.get_table("utterance")
        joined = utt_df.join(speaker_df, left_on="speaker_id", right_on="id", how="inner")

        if num_dictionaries > 1:
            subsets_per_dictionary = {}
            utts_per_dictionary = {}
            subsetted = 0
            # For each dictionary, filter utterances (joined with speaker info) to compute counts.
            for dict_name, dict_id in dictionary_lookup.items():
                base_df = joined.filter(pl.col("dictionary_id") == dict_id)
                base_df = add_filters(base_df)
                num_utts = base_df.height
                utts_per_dictionary[dict_name] = num_utts
                if num_utts < int(subset / num_dictionaries):
                    subsets_per_dictionary[dict_name] = num_utts
                    subsetted += 1

            remaining_subset = subset - sum(subsets_per_dictionary.values())
            remaining_dicts = num_dictionaries - subsetted
            remaining_subset_per_dictionary = int(remaining_subset / remaining_dicts) if remaining_dicts > 0 else 0

            for dict_name, num_utts in sorted(utts_per_dictionary.items(), key=lambda x: x[1]):
                dict_id = dictionary_lookup[dict_name]
                if dict_name in subsets_per_dictionary:
                    subset_per_dictionary = subsets_per_dictionary[dict_name]
                else:
                    subset_per_dictionary = remaining_subset_per_dictionary
                    remaining_dicts -= 1
                    if remaining_dicts > 0:
                        if num_utts < subset_per_dictionary:
                            remaining_subset -= num_utts
                        else:
                            remaining_subset -= subset_per_dictionary
                        remaining_subset_per_dictionary = int(remaining_subset / remaining_dicts) if remaining_dicts > 0 else 0

                logger.debug(f"For {dict_name}, total number of utterances is {num_utts}")
                larger_subset_num = int(subset_per_dictionary * 10)
                # Compute the average duration from the filtered utterances.
                base_df = joined.filter(pl.col("dictionary_id") == dict_id)
                base_df_f = add_filters(base_df)
                average_duration = base_df_f["duration"].mean() if base_df_f.height > 0 else None

                # Attempt to select candidate speaker IDs based on thresholds.
                speaker_ids = None
                for utt_count_cutoff in [30, 15, 5]:
                    df_cut = base_df_f.filter(pl.col("duration") <= average_duration) if average_duration is not None else base_df_f
                    if df_cut.height == 0:
                        continue
                    group = df_cut.group_by("speaker_id").agg(pl.count("id").alias("utt_count"))
                    valid = group.filter(pl.col("utt_count") >= utt_count_cutoff)
                    total_speaker_utterances = valid.select(pl.col("utt_count")).sum().item() if valid.height > 0 else 0
                    if total_speaker_utterances >= subset_per_dictionary:
                        speaker_ids = valid["speaker_id"].to_list()
                        break

                # Now, based on the total utterances, decide how to mark in_subset.
                if num_utts > larger_subset_num:
                    candidate_df = base_df_f
                    if speaker_ids is not None:
                        candidate_df = candidate_df.filter(pl.col("speaker_id").is_in(speaker_ids))
                    candidate_df = candidate_df.sort("duration").head(larger_subset_num)
                    subset_utts = (
                        candidate_df.sample(n=subset_per_dictionary, shuffle=True)
                        if candidate_df.height >= subset_per_dictionary else candidate_df
                    )
                    utt_df = self.polars_db.get_table("utterance")
                    utt_df = utt_df.with_columns(
                        pl.when(pl.col("id").is_in(subset_utts["id"]))
                        .then(True)
                        .otherwise(pl.col("in_subset"))
                        .alias("in_subset")
                    )
                    self.polars_db.replace_table("utterance", utt_df)
                    # Remove speakers with fewer than 5 utterances from the subset.
                    subset_group = subset_utts.group_by("speaker_id").agg(pl.count("id").alias("utt_count"))
                    remove_ids = subset_group.filter(pl.col("utt_count") < 5)["speaker_id"].to_list()
                    if remove_ids:
                        utt_df = self.polars_db.get_table("utterance")
                        utt_df = utt_df.with_columns(
                            pl.when(pl.col("speaker_id").is_in(remove_ids))
                            .then(False)
                            .otherwise(pl.col("in_subset"))
                            .alias("in_subset")
                        )
                        self.polars_db.replace_table("utterance", utt_df)
                    logger.debug(f"For {dict_name}, subset is {subset_per_dictionary}")
                elif num_utts > subset_per_dictionary:
                    candidate_df = base_df_f
                    if speaker_ids is not None:
                        candidate_df = candidate_df.filter(pl.col("speaker_id").is_in(speaker_ids))
                    subset_utts = (
                        candidate_df.sample(n=subset_per_dictionary, shuffle=True)
                        if candidate_df.height >= subset_per_dictionary else candidate_df
                    )
                    utt_df = self.polars_db.get_table("utterance")
                    utt_df = utt_df.with_columns(
                        pl.when(pl.col("id").is_in(subset_utts["id"]))
                        .then(True)
                        .otherwise(pl.col("in_subset"))
                        .alias("in_subset")
                    )
                    self.polars_db.replace_table("utterance", utt_df)
                    logger.debug(f"For {dict_name}, subset is {subset_per_dictionary}")
                else:
                    utt_ids = base_df_f["id"].to_list()
                    utt_df = self.polars_db.get_table("utterance")
                    utt_df = utt_df.with_columns(
                        pl.when(pl.col("id").is_in(utt_ids))
                        .then(True)
                        .otherwise(pl.col("in_subset"))
                        .alias("in_subset")
                    )
                    self.polars_db.replace_table("utterance", utt_df)

                # Reassign utterances for speakers with too few subset samples.
                group = base_df_f.group_by("speaker_id").agg(pl.count("id").alias("utt_count"))
                total_speaker_utterances = group.select(pl.col("utt_count")).sum().item() if group.height > 0 else 0
                remaining = subset_per_dictionary - total_speaker_utterances
                if remaining > 0:
                    speaker_ids_group = group["speaker_id"].to_list() if group.height > 0 else []
                    candidate_df = base_df_f.filter(pl.col("speaker_id").is_in(speaker_ids_group)) if speaker_ids_group else base_df_f
                    candidate_df = candidate_df.sort("duration").head(remaining * 10)
                    subset_utts = (
                        candidate_df.sample(n=remaining, shuffle=True)
                        if candidate_df.height >= remaining else candidate_df
                    )
                    utt_df = self.polars_db.get_table("utterance")
                    utt_df = utt_df.with_columns(
                        pl.when(pl.col("id").is_in(subset_utts["id"]))
                        .then(True)
                        .otherwise(pl.col("in_subset"))
                        .alias("in_subset")
                    )
                    self.polars_db.replace_table("utterance", utt_df)
        else:
            # Single dictionary case: simply mark a random sample of short utterances.
            larger_subset_num = subset * 10
            if subset < self.num_utterances:
                candidate_df = add_filters(utt_df).sort("duration").head(larger_subset_num)
                subset_utts = (
                    candidate_df.sample(n=subset, shuffle=True)
                    if candidate_df.height >= subset else candidate_df
                )
                utt_df = self.polars_db.get_table("utterance")
                utt_df = utt_df.with_columns(
                    pl.when(pl.col("id").is_in(subset_utts["id"]))
                    .then(True)
                    .otherwise(pl.col("in_subset"))
                    .alias("in_subset")
                )
                self.polars_db.replace_table("utterance", utt_df)
            else:
                utt_df = self.polars_db.get_table("utterance").with_columns(pl.lit(True).alias("in_subset"))
                self.polars_db.replace_table("utterance", utt_df)

        logger.debug(f"Setting subset flags took {time.time() - begin} seconds")

        # Create the output directory for the subset.
        subset_directory = self.corpus_output_directory.joinpath(f"subset_{subset}")
        log_dir = subset_directory.joinpath("log")
        os.makedirs(log_dir, exist_ok=True)

        # Determine which jobs have utterances in the subset.
        jobs_df = self.polars_db.get_table("job")
        utt_df = self.polars_db.get_table("utterance")
        job_utt_join = jobs_df.join(utt_df, left_on="id", right_on="job_id", how="inner")
        job_utt_join = job_utt_join.filter(pl.col("in_subset") == True)
        job_ids = job_utt_join.select("id").unique()["id"].to_list()
        jobs = [Job.from_dict(row) for row in jobs_df.filter(pl.col("id").is_in(job_ids)).to_dicts()]
        self._jobs = jobs

        arguments = [
            ExportKaldiFilesArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                None,
                subset_directory,
            )
            for j in self._jobs
        ]
        for _ in run_kaldi_function(ExportKaldiFilesFunction, arguments, total_count=subset):
            pass
        
    # @property
    # def num_files(self) -> int:
    #     """Number of files in the corpus"""
    #     if self._num_files is None:
    #         with self.session() as session:
    #             self._num_files = session.query(File).count()
    #     return self._num_files

    # @property
    # def num_utterances(self) -> int:
    #     """Number of utterances in the corpus"""
    #     if self._num_utterances is None:
    #         with self.session() as session:
    #             self._num_utterances = session.query(Utterance).count()
    #     return self._num_utterances

    # @property
    # def num_speakers(self) -> int:
    #     """Number of speakers in the corpus"""
    #     if self._num_speakers is None:
    #         with self.session() as session:
    #             self._num_speakers = session.query(sqlalchemy.func.count(Speaker.id)).scalar()
    #     return self._num_speakers

    # def subset_directory(self, subset: typing.Optional[int]) -> Path:
    #     """
    #     Construct a subset directory for the corpus

    #     Parameters
    #     ----------
    #     subset: int, optional
    #         Number of utterances to include, if larger than the total number of utterance or not specified, the
    #         split_directory is returned

    #     Returns
    #     -------
    #     str
    #         Path to subset directory
    #     """
    #     self._jobs = []
    #     with self.session() as session:
    #         c = session.query(Corpus).first()
    #         if subset is None or subset >= self.num_utterances or subset <= 0:
    #             c.current_subset = 0
    #         else:
    #             c.current_subset = subset
    #         session.commit()
    #     if subset is None or subset >= self.num_utterances or subset <= 0:
    #         if hasattr(self, "subset_lexicon"):
    #             self.subset_lexicon()
    #         return self.split_directory
    #     directory = self.corpus_output_directory.joinpath(f"subset_{subset}")
    #     if not os.path.exists(directory):
    #         self.create_subset(subset)
    #         if hasattr(self, "subset_lexicon"):
    #             self.subset_lexicon()
    #     return directory
    @property
    def num_files(self) -> int:
        """Number of files in the corpus"""
        if self._num_files is None:
            file_df = self.polars_db.get_table("file")
            self._num_files = file_df.height if not file_df.is_empty() else 0
        return self._num_files

    @property
    def num_utterances(self) -> int:
        """Number of utterances in the corpus"""
        if self._num_utterances is None:
            utt_df = self.polars_db.get_table("utterance")
            self._num_utterances = utt_df.height if not utt_df.is_empty() else 0
        return self._num_utterances

    @property
    def num_speakers(self) -> int:
        """Number of speakers in the corpus"""
        if self._num_speakers is None:
            spk_df = self.polars_db.get_table("speaker")
            self._num_speakers = spk_df.height if not spk_df.is_empty() else 0
        return self._num_speakers

    def subset_directory(self, subset: typing.Optional[int]) -> Path:
        """
        Construct a subset directory for the corpus.

        Parameters
        ----------
        subset : int, optional
            Number of utterances to include. If larger than the total number of utterances or not specified,
            the split_directory is returned.

        Returns
        -------
        Path
            Path to the subset directory.
        """
        # Reset jobs cache
        self._jobs = []

        # Retrieve the single corpus record from the in-memory DB.
        corpus_df = self.polars_db.get_table("corpus")
        if corpus_df.is_empty():
            raise Exception("Corpus not found in the in-memory DB.")
        corpus_dict = corpus_df.to_dicts()[0]
        corpus = Corpus.from_dict(corpus_dict)

        # Determine the current subset
        if subset is None or subset >= self.num_utterances or subset <= 0:
            corpus.current_subset = 0
        else:
            corpus.current_subset = subset
        # Update the corpus record in the in-memory DB.
        self.polars_db.replace_table("corpus", pl.DataFrame([corpus.to_dict()]))

        # If no valid subset is specified, run lexicon subsetting if available & return split_directory.
        if subset is None or subset >= self.num_utterances or subset <= 0:
            if hasattr(self, "subset_lexicon"):
                self.subset_lexicon()
            return self.split_directory

        # Create a new subset directory if necessary
        directory = self.corpus_output_directory.joinpath(f"subset_{subset}")
        if not os.path.exists(directory):
            self.create_subset(subset)
            if hasattr(self, "subset_lexicon"):
                self.subset_lexicon()
        return directory

    # def get_latest_workflow_run(self, workflow: WorkflowType, session: Session) -> CorpusWorkflow:
    #     """
    #     Get the latest version of a workflow type

    #     Parameters
    #     ----------
    #     workflow: :class:`~montreal_forced_aligner.data.WorkflowType`
    #         Workflow type
    #     session: :class:`sqlalchemy.orm.Session`
    #         Database session

    #     Returns
    #     -------
    #     :class:`~montreal_forced_aligner.db.CorpusWorkflow` or None
    #         Latest run of workflow type
    #     """
    #     workflow = (
    #         session.query(CorpusWorkflow)
    #         .filter(CorpusWorkflow.workflow_type == workflow)
    #         .order_by(CorpusWorkflow.time_stamp.desc())
    #         .first()
    #     )
    #     return workflow
    def get_latest_workflow_run(self, workflow: WorkflowType) -> CorpusWorkflow:
        """
        Get the latest version of a workflow type using the in-memory Polars DB.

        Parameters
        ----------
        workflow : WorkflowType
            Workflow type.

        Returns
        -------
        CorpusWorkflow or None
            Latest run of workflow type.
        """

        # Retrieve the corpus_workflow table as a Polars DataFrame.
        workflow_df = self.polars_db.get_table("corpus_workflow")
        if workflow_df.is_empty():
            return None

        # Filter rows with the matching workflow type.
        filtered_df = workflow_df.filter(pl.col("workflow_type") == workflow)
        if filtered_df.is_empty():
            return None

        # Sort by time_stamp in descending order and get the first record.
        latest_dict = filtered_df.sort("time_stamp", descending=True).to_dicts()[0]
        return CorpusWorkflow.from_dict(latest_dict)

    def _load_corpus(self) -> None:
        """
        Load the corpus
        """
        self.inspect_database()
        logger.info("Setting up corpus information...")
        if not self.imported:
            logger.debug("Could not load from temp")
            logger.info("Loading corpus from source files...")
            if config.USE_MP:
                self._load_corpus_from_source_mp()
            else:
                self._load_corpus_from_source()
        else:
            logger.debug("Successfully loaded from temporary files")
        if not self.num_files:
            raise CorpusError(
                "There were no files found for this corpus. Please validate the corpus."
            )
        if not self.num_speakers:
            raise CorpusError(
                "There were no sound files found of the appropriate format. Please double check the corpus path "
                "and/or run the validation utility (mfa validate)."
            )
        average_utterances = self.num_utterances / self.num_speakers
        logger.info(
            f"Found {self.num_speakers} speaker{'s' if self.num_speakers > 1 else ''} across {self.num_files} file{'s' if self.num_files > 1 else ''}, "
            f"average number of utterances per speaker: {average_utterances}"
        )

    @property
    def base_data_directory(self) -> str:
        """Corpus data directory"""
        return self.corpus_output_directory

    @property
    def data_directory(self) -> str:
        """Corpus data directory"""
        return self.split_directory

    @abstractmethod
    def _load_corpus_from_source_mp(self) -> None:
        """Abstract method for loading a corpus with multiprocessing"""
        ...

    @abstractmethod
    def _load_corpus_from_source(self) -> None:
        """Abstract method for loading a corpus without multiprocessing"""
        ...