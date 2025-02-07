"""Class definitions for base aligner"""
from __future__ import annotations

import collections
import csv
import functools
import io
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import typing
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional

import polars as pl
from kalpy.feat.mfcc import MfccComputer
from kalpy.feat.pitch import PitchComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.utils import read_transition_model
from sqlalchemy.orm import joinedload, subqueryload
from tqdm.rich import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.abc import FileExporterMixin
from montreal_forced_aligner.alignment.mixins import AlignMixin
from montreal_forced_aligner.alignment.multiprocessing import (
    AlignmentExtractionArguments,
    AlignmentExtractionFunction,
    AnalyzeAlignmentsArguments,
    AnalyzeAlignmentsFunction,
    ExportTextGridArguments,
    ExportTextGridProcessWorker,
    FineTuneArguments,
    FineTuneFunction,
    GeneratePronunciationsArguments,
    GeneratePronunciationsFunction,
)
from montreal_forced_aligner.corpus.acoustic_corpus import AcousticCorpusPronunciationMixin
from montreal_forced_aligner.data import (
    CtmInterval,
    PhoneType,
    PronunciationProbabilityCounter,
    TextFileType,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.db_polars import (
    Corpus,
    CorpusWorkflow,
    Dictionary,
    File,
    Phone,
    PhoneInterval,
    PhonologicalRule,
    Pronunciation,
    RuleApplication,
    SoundFile,
    Speaker,
    TextFile,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignmentExportError, KaldiProcessingError
from montreal_forced_aligner.helper import (
    align_phones,
    format_correction,
    format_probability,
    mfa_open,
)
from montreal_forced_aligner.textgrid import (
    construct_textgrid_output,
    output_textgrid_writing_errors,
)
from montreal_forced_aligner.utils import log_kaldi_errors, run_kaldi_function

if typing.TYPE_CHECKING:
    from montreal_forced_aligner.abc import MetaDict

__all__ = ["CorpusAligner"]


logger = logging.getLogger("mfa")


class CorpusAligner(AcousticCorpusPronunciationMixin, AlignMixin, FileExporterMixin):
    """
    Mixin class that aligns corpora with pronunciation dictionaries

    See Also
    --------
    :class:`~montreal_forced_aligner.corpus.acoustic_corpus.AcousticCorpusPronunciationMixin`
        For dictionary and corpus parsing parameters
    :class:`~montreal_forced_aligner.alignment.mixins.AlignMixin`
        For alignment parameters
    :class:`~montreal_forced_aligner.abc.FileExporterMixin`
        For file exporting parameters
    """

    def __init__(
        self, g2p_model_path: Path = None, max_active: int = 2500, lattice_beam: int = 6, **kwargs
    ):
        super().__init__(**kwargs)
        self.export_output_directory = None
        self.max_active = max_active
        self.lattice_beam = lattice_beam
        self.phone_lm_order = 2
        self.phone_lm_method = "unsmoothed"
        self.alignment_mode = True
        self.g2p_model = None
        if g2p_model_path:
            from montreal_forced_aligner.models import G2PModel

            if isinstance(g2p_model_path, dict):
                self.g2p_model = {k: G2PModel(v) for k, v in g2p_model_path.items()}
            else:
                self.g2p_model = G2PModel(g2p_model_path)

    @property
    def hclg_options(self) -> MetaDict:
        """Options for constructing HCLG FSTs"""
        return {
            "self_loop_scale": self.self_loop_scale,
            "transition_scale": self.transition_scale,
        }

    @property
    def decode_options(self) -> MetaDict:
        """Options needed for decoding"""
        return {
            "beam": self.beam,
            "max_active": self.max_active,
            "lattice_beam": self.lattice_beam,
            "acoustic_scale": self.acoustic_scale,
        }

    @property
    def score_options(self) -> MetaDict:
        """Options needed for scoring lattices"""
        return {
            "frame_shift": round(self.frame_shift / 1000, 3),
            "acoustic_scale": self.acoustic_scale,
            "language_model_weight": getattr(self, "language_model_weight", 10),
            "word_insertion_penalty": getattr(self, "word_insertion_penalty", 0.5),
        }

    def analyze_alignments_arguments(self) -> List[AnalyzeAlignmentsArguments]:
        return [
            AnalyzeAlignmentsArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"calculate_speech_post.{j.id}.log"),
                self.model_path,
                self.align_options,
            )
            for j in self.jobs
        ]

    # def analyze_alignments(self):
    #     if not config.USE_POSTGRES:
    #         logger.warning("Alignment analysis not available without using postgresql")
    #         return
    #     workflow = self.current_workflow
    #     if not workflow.alignments_collected:
    #         self.collect_alignments()
    #     logger.info("Analyzing alignment quality...")
    #     begin = time.time()
    #     with self.session() as session:
    #         update_mappings = []
    #         query = session.query(
    #             PhoneInterval.phone_id,
    #             sqlalchemy.func.avg(PhoneInterval.duration),
    #             sqlalchemy.func.stddev_samp(PhoneInterval.duration),
    #         ).group_by(PhoneInterval.phone_id)
    #         for p_id, mean_duration, sd_duration in query:
    #             update_mappings.append(
    #                 {"id": p_id, "mean_duration": mean_duration, "sd_duration": sd_duration}
    #             )
    #         bulk_update(session, Phone, update_mappings)
    #         session.commit()

    #         arguments = self.analyze_alignments_arguments()
    #         update_mappings = []
    #         for utt_id, speech_log_likelihood, duration_deviation in run_kaldi_function(
    #             AnalyzeAlignmentsFunction, arguments, total_count=self.num_current_utterances
    #         ):
    #             update_mappings.append(
    #                 {
    #                     "id": utt_id,
    #                     "speech_log_likelihood": speech_log_likelihood,
    #                     "duration_deviation": duration_deviation,
    #                 }
    #             )

    #         bulk_update(session, Utterance, update_mappings)
    #         session.commit()

    #         csv_path = self.working_directory.joinpath("alignment_analysis.csv")
    #         with mfa_open(csv_path, "w") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(
    #                 [
    #                     "file",
    #                     "begin",
    #                     "end",
    #                     "speaker",
    #                     "overall_log_likelihood",
    #                     "speech_log_likelihood",
    #                     "phone_duration_deviation",
    #                 ]
    #             )
    #             utterances = (
    #                 session.query(
    #                     File.name,
    #                     Utterance.begin,
    #                     Utterance.end,
    #                     Speaker.name,
    #                     Utterance.alignment_log_likelihood,
    #                     Utterance.speech_log_likelihood,
    #                     Utterance.duration_deviation,
    #                 )
    #                 .join(Utterance.file)
    #                 .join(Utterance.speaker)
    #             )
    #             for row in utterances:
    #                 writer.writerow([*row])
    #     logger.debug(f"Analyzed alignment quality in {time.time() - begin:.3f} seconds")
    def analyze_alignments(self):
        """
        Analyze the alignment quality using in-memory Polars dataframes.
        
        Instead of using a database session, we now use the PolarsDB instance available on self.db.
        This calculates averages and standard deviations of phone durations,
        updates the Phone and Utterance tables, and produces an output CSV by
        joining the File, Utterance, and Speaker tables.
        """

        logger.info("Analyzing alignment quality...")
        begin = time.time()

        # Ensure the workflow has collected alignments
        workflow = self.current_workflow
        if not workflow.alignments_collected:
            self.collect_alignments()

        # Assume self.db is the PolarsDB instance (see db_polars.py)
        db = self.polars_db

        # --- Update Phone table ---
        # Get phone intervals and compute the duration as (end - begin)
        phone_intervals = db.get_table("phone_interval")
        if phone_intervals.is_empty():
            logger.warning("No phone intervals found for analysis.")
        else:
            # Compute duration if not already present
            phone_intervals = phone_intervals.with_columns((pl.col("end") - pl.col("begin")).alias("duration"))
            # Group by phone_id and compute average and standard deviation of durations
            grouped = phone_intervals.groupby("phone_id").agg([
                pl.col("duration").mean().alias("mean_duration"),
                pl.col("duration").std().alias("sd_duration")
            ])
            # Create update mappings for the Phone table (convert phone_id to id)
            update_mappings = [
                {"id": row["phone_id"], "mean_duration": row["mean_duration"], "sd_duration": row["sd_duration"]}
                for row in grouped.to_dicts()
            ]
            db.bulk_update("phone", update_mappings)

        # --- Update Utterance table ---
        # Get updated arguments (perhaps produced based on the current workflow)
        arguments = self.analyze_alignments_arguments()
        # Use a list comprehension to generate update mappings in one pass
        update_mappings = [
            {"id": utt_id, "speech_log_likelihood": speech_log_likelihood, "duration_deviation": duration_deviation}
            for utt_id, speech_log_likelihood, duration_deviation in run_kaldi_function(
                AnalyzeAlignmentsFunction, arguments, total_count=self.num_current_utterances
            )
        ]
        db.bulk_update("utterance", update_mappings)

        # --- Generate CSV output ---
        # Instead of SQL joins, perform in-memory joins using Polars
        df_file = db.get_table("file")
        df_utterance = db.get_table("utterance")
        df_speaker = db.get_table("speaker")

        if df_file.is_empty() or df_utterance.is_empty() or df_speaker.is_empty():
            logger.warning("Insufficient data in file, utterance, or speaker tables for CSV export.")
        else:
            # Join utterance with file on file_id = file.id
            df_utt_file = df_utterance.join(df_file, left_on="file_id", right_on="id", how="left")
            # Join the above result with speaker on speaker_id = speaker.id
            df_joined = df_utt_file.join(df_speaker, left_on="speaker_id", right_on="id", how="left", suffix="_spk")
            # Select exactly the columns required for the output CSV.
            # Note: 'name' from file is kept as the file name while the speaker name comes from speaker.name
            df_csv = df_joined.select([
                pl.col("name").alias("file"),
                pl.col("begin"),
                pl.col("end"),
                pl.col("name_spk").alias("speaker"),
                pl.col("alignment_log_likelihood"),
                pl.col("speech_log_likelihood"),
                pl.col("duration_deviation")
            ])
            csv_path = self.working_directory.joinpath("alignment_analysis.csv")
            # Write out CSV using Polars write_csv
            df_csv.write_csv(str(csv_path))

        logger.debug(f"Analyzed alignment quality in {time.time() - begin:.3f} seconds")

    def alignment_extraction_arguments(self) -> List[AlignmentExtractionArguments]:
        """
        Generate Job arguments for
        :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionArguments`]
            Arguments for processing
        """
        arguments = []
        workflow = self.current_workflow
        from_transcription = False
        if workflow.workflow_type in (
            WorkflowType.per_speaker_transcription,
            WorkflowType.transcription,
            WorkflowType.phone_transcription,
        ):
            from_transcription = True

        transition_model = read_transition_model(str(self.alignment_model_path))
        lexicon_compilers = {}
        if getattr(self, "use_g2p", False):
            lexicon_compilers = getattr(self, "lexicon_compilers", {})
        for j in self.jobs:
            arguments.append(
                AlignmentExtractionArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    self.working_log_directory.joinpath(f"get_phone_ctm.{j.id}.log"),
                    self.working_directory,
                    lexicon_compilers,
                    transition_model,
                    round(self.frame_shift / 1000, 4),
                    self.score_options,
                    self.phone_confidence,
                    from_transcription,
                    self.use_g2p,
                )
            )

        return arguments

    def export_textgrid_arguments(
        self, output_format: str, include_original_text: bool = False
    ) -> List[ExportTextGridArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`

        Parameters
        ----------
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridArguments`]
            Arguments for processing
        """
        return [
            ExportTextGridArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"export_textgrids.{j.id}.log"),
                self.export_frame_shift,
                config.CLEANUP_TEXTGRIDS,
                self.clitic_marker,
                self.export_output_directory,
                output_format,
                include_original_text,
            )
            for j in self.jobs
        ]

    def generate_pronunciations_arguments(
        self,
    ) -> List[GeneratePronunciationsArguments]:
        """
        Generate Job arguments for :func:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsArguments`]
            Arguments for processing
        """
        align_options = self.align_options
        align_options.pop("boost_silence", 1.0)
        disambiguation_symbols = [self.phone_mapping[p] for p in self.disambiguation_symbols]
        aligner = GmmAligner(
            self.model_path, disambiguation_symbols=disambiguation_symbols, **align_options
        )
        lexicon_compilers = {}
        if getattr(self, "use_g2p", False):
            lexicon_compilers = getattr(self, "lexicon_compilers", {})
        return [
            GeneratePronunciationsArguments(
                j.id,
                getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                self.working_log_directory.joinpath(f"generate_pronunciations.{j.id}.log"),
                aligner,
                lexicon_compilers,
                False,
            )
            for j in self.jobs
        ]

    # def align(self, workflow_name=None) -> None:
    #     """Run the aligner"""
    #     self.alignment_mode = True
    #     self.initialize_database()
    #     wf = self.current_workflow
    #     if wf is None:
    #         self.create_new_current_workflow(WorkflowType.alignment, workflow_name)
    #     wf = self.current_workflow
    #     if wf.done:
    #         logger.info("Alignment already done, skipping.")
    #         return
    #     begin = time.time()
    #     acoustic_model = getattr(self, "acoustic_model", None)
    #     if acoustic_model is not None:
    #         acoustic_model.export_model(self.working_directory)
    #     perform_speaker_adaptation = self.uses_speaker_adaptation and not config.SINGLE_SPEAKER
    #     final_alignment = self.final_alignment
    #     if perform_speaker_adaptation:
    #         self.final_alignment = False
    #     try:
    #         self.uses_speaker_adaptation = False

    #         self.compile_train_graphs()

    #         logger.info("Performing first-pass alignment...")
    #         for j in self.jobs:
    #             paths = j.construct_path_dictionary(self.working_directory, "trans", "ark")
    #             for p in paths.values():
    #                 if os.path.exists(p):
    #                     os.remove(p)

    #         self.align_utterances()
    #         if (
    #             acoustic_model is not None
    #             and acoustic_model.meta["features"]["uses_speaker_adaptation"]
    #             and perform_speaker_adaptation
    #         ):
    #             self.calc_fmllr()
    #             if final_alignment:
    #                 self.final_alignment = True
    #             self.uses_speaker_adaptation = True
    #             assert self.alignment_model_path.suffix == ".mdl"
    #             logger.info("Performing second-pass alignment...")
    #             self.align_utterances()
    #         self.collect_alignments()
    #         if self.use_phone_model:
    #             self.transcribe(WorkflowType.phone_transcription)
    #         elif self.fine_tune:
    #             self.fine_tune_alignments()

    #         with self.session() as session:
    #             session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
    #                 {"done": True}
    #             )
    #             session.commit()
    #     except Exception as e:
    #         with self.session() as session:
    #             session.query(CorpusWorkflow).filter(CorpusWorkflow.id == wf.id).update(
    #                 {"dirty": True}
    #             )
    #             session.commit()
    #         if isinstance(e, KaldiProcessingError):
    #             log_kaldi_errors(e.error_logs)
    #             e.update_log_file()
    #         raise
    #     logger.debug(f"Generated alignments in {time.time() - begin:.3f} seconds")
    def align(self, workflow_name=None) -> None:
        """Run the aligner using the in‑memory Polars database."""

        logger.info("Starting alignment mode...")
        self.alignment_mode = True
        self.initialize_database()
        wf = self.current_workflow
        if wf is None:
            self.create_new_current_workflow(WorkflowType.alignment, workflow_name)
        wf = self.current_workflow
        if wf.done:
            logger.info("Alignment already done, skipping.")
            return
        begin = time.time()

        acoustic_model = getattr(self, "acoustic_model", None)
        if acoustic_model is not None:
            acoustic_model.export_model(self.working_directory)

        perform_speaker_adaptation = self.uses_speaker_adaptation and not config.SINGLE_SPEAKER
        final_alignment = self.final_alignment
        if perform_speaker_adaptation:
            self.final_alignment = False

        try:
            self.uses_speaker_adaptation = False
            self.compile_train_graphs()

            logger.info("Performing first-pass alignment...")
            # Remove any pre‑existing transcription files using a try/except to avoid redundant existence checks.
            for j in self.jobs:
                for p in j.construct_path_dictionary(self.working_directory, "trans", "ark").values():
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            self.align_utterances()

            if (acoustic_model is not None and
                acoustic_model.meta["features"]["uses_speaker_adaptation"] and
                perform_speaker_adaptation):
                self.calc_fmllr()
                if final_alignment:
                    self.final_alignment = True
                self.uses_speaker_adaptation = True
                assert self.alignment_model_path.suffix == ".mdl"
                logger.info("Performing second-pass alignment...")
                self.align_utterances()

            self.collect_alignments()

            if self.use_phone_model:
                self.transcribe(WorkflowType.phone_transcription)
            elif self.fine_tune:
                self.fine_tune_alignments()

            # Update the workflow record in the Polars-based database to mark it "done".
            self.polars_db.bulk_update("corpus_workflow", [{"id": wf.id, "done": True}])
        except Exception as e:
            # In case of error, mark the workflow as "dirty" in the Polars database.
            self.polars_db.bulk_update("corpus_workflow", [{"id": wf.id, "dirty": True}])
            if isinstance(e, KaldiProcessingError):
                log_kaldi_errors(e.error_logs)
                e.update_log_file()
            raise

        logger.debug(f"Generated alignments in {time.time() - begin:.3f} seconds")

    # def compute_pronunciation_probabilities(self):
    #     """
    #     Multiprocessing function that computes pronunciation probabilities from alignments

    #     See Also
    #     --------
    #     :class:`~montreal_forced_aligner.alignment.multiprocessing.GeneratePronunciationsFunction`
    #         Multiprocessing helper function for each job
    #     :meth:`.CorpusAligner.generate_pronunciations_arguments`
    #         Job method for generating arguments for the helper function
    #     :kaldi_steps:`align_si`
    #         Reference Kaldi script
    #     :kaldi_steps:`align_fmllr`
    #         Reference Kaldi script
    #     """

    #     begin = time.time()
    #     with self.session() as session:
    #         dictionary_counters = {
    #             dict_id: PronunciationProbabilityCounter()
    #             for dict_id, in session.query(Dictionary.id).filter(Dictionary.name != "default")
    #         }
    #     logger.info("Generating pronunciations...")
    #     arguments = self.generate_pronunciations_arguments()
    #     for result in run_kaldi_function(
    #         GeneratePronunciationsFunction, arguments, total_count=self.num_current_utterances
    #     ):
    #         try:
    #             dict_id, utterance_counter = result
    #             dictionary_counters[dict_id].add_counts(utterance_counter)
    #         except Exception:
    #             import sys
    #             import traceback

    #             exc_type, exc_value, exc_traceback = sys.exc_info()
    #             print("\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    #             raise
    #     initial_key = ("<s>", "")
    #     final_key = ("</s>", "")
    #     lambda_2 = 2
    #     silence_prob_sum = 0
    #     initial_silence_prob_sum = 0
    #     final_silence_correction_sum = 0
    #     final_non_silence_correction_sum = 0
    #     with mfa_open(
    #         self.working_log_directory.joinpath("pronunciation_probability_calculation.log"),
    #         "w",
    #         encoding="utf8",
    #     ) as log_file, self.session() as session:
    #         session.query(Pronunciation).update({"count": 0})
    #         session.commit()
    #         dictionaries = session.query(Dictionary)
    #         dictionary_mappings = []
    #         for d in dictionaries:
    #             d_id = d.id
    #             if d_id not in dictionary_counters:
    #                 continue
    #             counter = dictionary_counters[d_id]
    #             log_file.write(f"For {d.name}:\n")
    #             words = (
    #                 session.query(Word.word)
    #                 .filter(Word.dictionary_id == d_id)
    #                 .filter(
    #                     sqlalchemy.or_(
    #                         sqlalchemy.and_(
    #                             Word.word_type.in_(WordType.speech_types()), Word.count > 0
    #                         ),
    #                         Word.word.in_(
    #                             [d.cutoff_word, d.oov_word, d.laughter_word, d.bracketed_word]
    #                         ),
    #                     )
    #                 )
    #             )
    #             pronunciations = (
    #                 session.query(
    #                     Word.word,
    #                     Pronunciation.pronunciation,
    #                     Pronunciation.id,
    #                     Pronunciation.generated_by_rule,
    #                 )
    #                 .join(Pronunciation.word)
    #                 .filter(Word.dictionary_id == d_id)
    #                 .filter(
    #                     sqlalchemy.or_(
    #                         sqlalchemy.and_(
    #                             Word.word_type.in_(WordType.speech_types()), Word.count > 0
    #                         ),
    #                         Word.word.in_(
    #                             [d.cutoff_word, d.oov_word, d.laughter_word, d.bracketed_word]
    #                         ),
    #                     )
    #                 )
    #             )
    #             pron_mapping = {}
    #             pronunciations = [
    #                 (w, p, p_id)
    #                 for w, p, p_id, generated in pronunciations
    #                 if not generated or p in counter.word_pronunciation_counts[w]
    #             ]
    #             pronunciations.append((d.cutoff_word, "cutoff_model", None))
    #             for w, p, p_id in pronunciations:
    #                 pron_mapping[(w, p)] = {"id": p_id}
    #                 if w in {initial_key[0], final_key[0], self.silence_word}:
    #                     continue
    #                 counter.word_pronunciation_counts[w][p] += 1  # Add one smoothing
    #             for (w,) in words:
    #                 if w in {initial_key[0], final_key[0], self.silence_word}:
    #                     continue
    #                 if w not in counter.word_pronunciation_counts:
    #                     continue
    #                 pron_counts = counter.word_pronunciation_counts[w]
    #                 max_value = max(pron_counts.values())
    #                 for p, c in pron_counts.items():
    #                     if self.position_dependent_phones:
    #                         p = re.sub(r"_[BSEI]\b", "", p)
    #                     if (w, p) in pron_mapping:
    #                         pron_mapping[(w, p)]["count"] = c
    #                         pron_mapping[(w, p)]["probability"] = format_probability(c / max_value)

    #             silence_count = sum(counter.silence_before_counts.values())
    #             non_silence_count = sum(counter.non_silence_before_counts.values())
    #             log_file.write(f"Total silence count was {silence_count}\n")
    #             log_file.write(f"Total non silence count was {non_silence_count}\n")
    #             silence_probability = format_probability(
    #                 silence_count / (silence_count + non_silence_count)
    #             )
    #             silence_prob_sum += silence_probability
    #             silence_probabilities = {}
    #             for w, p, _ in pronunciations:
    #                 count = counter.silence_following_counts[(w, p)]
    #                 total_count = (
    #                     counter.silence_following_counts[(w, p)]
    #                     + counter.non_silence_following_counts[(w, p)]
    #                 )
    #                 pron_mapping[(w, p)]["silence_following_count"] = count
    #                 pron_mapping[(w, p)][
    #                     "non_silence_following_count"
    #                 ] = counter.non_silence_following_counts[(w, p)]
    #                 w_p_silence_count = count + (silence_probability * lambda_2)
    #                 prob = format_probability(w_p_silence_count / (total_count + lambda_2))
    #                 silence_probabilities[(w, p)] = prob
    #                 if w not in {initial_key[0], final_key[0], self.silence_word}:
    #                     pron_mapping[(w, p)]["silence_after_probability"] = prob
    #             lambda_3 = 2
    #             bar_count_silence_wp = collections.defaultdict(float)
    #             bar_count_non_silence_wp = collections.defaultdict(float)
    #             for (w_p1, w_p2), counts in counter.ngram_counts.items():
    #                 if w_p1 not in silence_probabilities:
    #                     silence_prob = 0.01
    #                 else:
    #                     silence_prob = silence_probabilities[w_p1]
    #                 total_count = counts["silence"] + counts["non_silence"]
    #                 bar_count_silence_wp[w_p2] += total_count * silence_prob
    #                 bar_count_non_silence_wp[w_p2] += total_count * (1 - silence_prob)
    #             for w, p, _ in pronunciations:
    #                 if w in {initial_key[0], final_key[0], self.silence_word}:
    #                     continue
    #                 silence_count = counter.silence_before_counts[(w, p)]
    #                 non_silence_count = counter.non_silence_before_counts[(w, p)]
    #                 pron_mapping[(w, p)]["silence_before_correction"] = format_correction(
    #                     (silence_count + lambda_3) / (bar_count_silence_wp[(w, p)] + lambda_3)
    #                 )

    #                 pron_mapping[(w, p)]["non_silence_before_correction"] = format_correction(
    #                     (non_silence_count + lambda_3)
    #                     / (bar_count_non_silence_wp[(w, p)] + lambda_3)
    #                 )
    #             cutoff_model = pron_mapping.pop((d.cutoff_word, "cutoff_model"))
    #             bulk_update(session, Pronunciation, list(pron_mapping.values()))
    #             session.flush()
    #             cutoff_not_model = pron_mapping[(d.cutoff_word, "spn")]
    #             cutoff_query = (
    #                 session.query(Pronunciation.id, Pronunciation.pronunciation)
    #                 .join(Pronunciation.word)
    #                 .filter(Word.word != d.cutoff_word)
    #                 .filter(Word.word.like(f"{d.cutoff_word[:-1]}%"))
    #             )
    #             cutoff_mappings = []
    #             for pron_id, pron in cutoff_query:
    #                 if pron == d.cutoff_word:
    #                     data = dict(cutoff_not_model)
    #                 else:
    #                     data = dict(cutoff_model)
    #                 data["id"] = pron_id
    #                 cutoff_mappings.append(data)
    #             if cutoff_mappings:
    #                 bulk_update(session, Pronunciation, cutoff_mappings)
    #                 session.flush()

    #             initial_silence_count = counter.silence_before_counts[initial_key] + (
    #                 silence_probability * lambda_2
    #             )
    #             initial_non_silence_count = counter.non_silence_before_counts[initial_key] + (
    #                 (1 - silence_probability) * lambda_2
    #             )
    #             initial_silence_probability = format_probability(
    #                 initial_silence_count / (initial_silence_count + initial_non_silence_count)
    #             )

    #             final_silence_correction = format_correction(
    #                 (counter.silence_before_counts[final_key] + lambda_3)
    #                 / (bar_count_silence_wp[final_key] + lambda_3)
    #             )

    #             final_non_silence_correction = format_correction(
    #                 (counter.non_silence_before_counts[final_key] + lambda_3)
    #                 / (bar_count_non_silence_wp[final_key] + lambda_3)
    #             )
    #             initial_silence_prob_sum += initial_silence_probability
    #             final_silence_correction_sum += final_silence_correction
    #             final_non_silence_correction_sum += final_non_silence_correction
    #             dictionary_mappings.append(
    #                 {
    #                     "id": d_id,
    #                     "silence_probability": silence_probability,
    #                     # "initial_silence_probability": initial_silence_probability,
    #                     # "final_silence_correction": final_silence_correction,
    #                     # "final_non_silence_correction": final_non_silence_correction,
    #                 }
    #             )

    #         self.silence_probability = format_probability(silence_prob_sum / self.num_dictionaries)
    #         self.initial_silence_probability = format_probability(
    #             initial_silence_prob_sum / self.num_dictionaries
    #         )
    #         self.final_silence_correction = format_probability(
    #             final_silence_correction_sum / self.num_dictionaries
    #         )
    #         self.final_non_silence_correction = (
    #             final_non_silence_correction_sum / self.num_dictionaries
    #         )
    #         bulk_update(session, Dictionary, dictionary_mappings)
    #         session.commit()
    #         rules: List[PhonologicalRule] = (
    #             session.query(PhonologicalRule)
    #             .options(
    #                 subqueryload(PhonologicalRule.pronunciations).joinedload(
    #                     RuleApplication.pronunciation, innerjoin=True
    #                 )
    #             )
    #             .all()
    #         )
    #         if rules:
    #             rules_for_deletion = []
    #             for r in rules:
    #                 base_count = 0
    #                 base_sil_after_count = 0
    #                 base_nonsil_after_count = 0

    #                 rule_count = 0
    #                 rule_sil_before_correction = 0
    #                 base_sil_before_correction = 0
    #                 rule_nonsil_before_correction = 0
    #                 base_nonsil_before_correction = 0
    #                 rule_sil_after_count = 0
    #                 rule_nonsil_after_count = 0
    #                 rule_correction_count = 0
    #                 base_correction_count = 0
    #                 non_application_query = session.query(Pronunciation).filter(
    #                     Pronunciation.pronunciation.regexp_match(
    #                         r.match_regex.pattern.replace("?P<segment>", "")
    #                         .replace("?P<preceding>", "")
    #                         .replace("?P<following>", "")
    #                     ),
    #                     Pronunciation.count > 1,
    #                 )
    #                 for p in non_application_query:
    #                     base_count += p.count

    #                     if p.silence_before_correction:
    #                         base_sil_before_correction += p.silence_before_correction
    #                         base_nonsil_before_correction += p.non_silence_before_correction
    #                         base_correction_count += 1

    #                     base_sil_after_count += (
    #                         p.silence_following_count if p.silence_following_count else 0
    #                     )
    #                     base_nonsil_after_count += (
    #                         p.non_silence_following_count if p.non_silence_following_count else 0
    #                     )

    #                 for p in r.pronunciations:
    #                     p = p.pronunciation
    #                     rule_count += p.count

    #                     if p.silence_before_correction:
    #                         rule_sil_before_correction += p.silence_before_correction
    #                         rule_nonsil_before_correction += p.non_silence_before_correction
    #                         rule_correction_count += 1

    #                     rule_sil_after_count += (
    #                         p.silence_following_count if p.silence_following_count else 0
    #                     )
    #                     rule_nonsil_after_count += (
    #                         p.non_silence_following_count if p.non_silence_following_count else 0
    #                     )
    #                 if not rule_count:
    #                     rules_for_deletion.append(r)
    #                     continue
    #                 r.probability = format_probability(rule_count / (rule_count + base_count))
    #                 if rule_correction_count:
    #                     rule_sil_before_correction = (
    #                         rule_sil_before_correction / rule_correction_count
    #                     )
    #                     rule_nonsil_before_correction = (
    #                         rule_nonsil_before_correction / rule_correction_count
    #                     )
    #                     if base_correction_count:
    #                         base_sil_before_correction = (
    #                             base_sil_before_correction / base_correction_count
    #                         )
    #                         base_nonsil_before_correction = (
    #                             base_nonsil_before_correction / base_correction_count
    #                         )
    #                     else:
    #                         base_sil_before_correction = 1.0
    #                         base_nonsil_before_correction = 1.0
    #                     r.silence_before_correction = format_correction(
    #                         rule_sil_before_correction - base_sil_before_correction,
    #                         positive_only=False,
    #                     )
    #                     r.non_silence_before_correction = format_correction(
    #                         rule_nonsil_before_correction - base_nonsil_before_correction,
    #                         positive_only=False,
    #                     )

    #                 silence_after_probability = format_probability(
    #                     (rule_sil_after_count + lambda_2)
    #                     / (rule_sil_after_count + rule_nonsil_after_count + lambda_2)
    #                 )
    #                 base_sil_after_probability = format_probability(
    #                     (base_sil_after_count + lambda_2)
    #                     / (base_sil_after_count + base_nonsil_after_count + lambda_2)
    #                 )
    #                 r.silence_after_probability = format_correction(
    #                     silence_after_probability / base_sil_after_probability
    #                 )
    #             previous_pronunciation_counts = {
    #                 k: v
    #                 for k, v in session.query(
    #                     Dictionary.name, sqlalchemy.func.count(Pronunciation.id)
    #                 )
    #                 .join(Pronunciation.word)
    #                 .join(Word.dictionary)
    #                 .group_by(Dictionary.name)
    #             }
    #             for r in rules_for_deletion:
    #                 logger.debug(f"Removing {r} for zero counts.")
    #             session.query(RuleApplication).filter(
    #                 RuleApplication.rule_id.in_([r.id for r in rules_for_deletion])
    #             ).delete()
    #             session.flush()
    #             session.query(PhonologicalRule).filter(
    #                 PhonologicalRule.id.in_([r.id for r in rules_for_deletion])
    #             ).delete()
    #             session.flush()
    #             session.query(Pronunciation).filter(
    #                 Pronunciation.count == 0, Pronunciation.generated_by_rule == True  # noqa
    #             ).delete()
    #             session.commit()
    #             pronunciation_counts = {
    #                 k: v
    #                 for k, v in session.query(
    #                     Dictionary.name, sqlalchemy.func.count(Pronunciation.id)
    #                 )
    #                 .join(Pronunciation.word)
    #                 .join(Word.dictionary)
    #                 .group_by(Dictionary.name)
    #             }
    #             for d_name, c in pronunciation_counts.items():
    #                 prev_c = previous_pronunciation_counts[d_name]
    #                 logger.debug(
    #                     f"{d_name}: Reduced number of pronunciations from {prev_c} to {c}"
    #                 )
    #     logger.debug(
    #         f"Calculating pronunciation probabilities took {time.time() - begin:.3f} seconds"
    #     )
    def compute_pronunciation_probabilities(self):
        """
        Multiprocessing function that computes pronunciation probabilities from alignments
        using an in‑memory Polars database. This version replaces all SQLAlchemy sessions with
        Polars operations and makes use of vectorized updates to optimize the computations.
        
        Assumes that:
        - self.db is an instance of PolarsDB.
        - Helper functions like format_probability(), format_correction(), and the
            PronunciationProbabilityCounter class are available.
        - Tables "dictionary", "word", "pronunciation", "rule_applications" and "phonological_rule"
            exist in self.db.
        """

        begin = time.time()
        
        # --- Setup dictionary counters ---
        # Get non-default dictionaries
        dict_df = self.polars_db.get_table("dictionary")
        non_default_dicts = dict_df.filter(pl.col("name") != "default")
        # Create a counter for each non-default dictionary using its id.
        dictionary_counters = {
            d["id"]: PronunciationProbabilityCounter()
            for d in non_default_dicts.to_dicts()
        }
        
        logger.info("Generating pronunciations...")
        arguments = self.generate_pronunciations_arguments()
        for result in run_kaldi_function(
            GeneratePronunciationsFunction, arguments, total_count=self.num_current_utterances
        ):
            try:
                dict_id, utterance_counter = result
                dictionary_counters[dict_id].add_counts(utterance_counter)
            except Exception:
                import sys, traceback

                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("\n".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
                raise

        # Setup common variables
        initial_key = ("<s>", "")
        final_key = ("</s>", "")
        lambda_2 = 2
        silence_prob_sum = 0
        initial_silence_prob_sum = 0
        final_silence_correction_sum = 0
        final_non_silence_correction_sum = 0

        # --- Open log file and process each dictionary ---
        log_path = self.working_log_directory.joinpath("pronunciation_probability_calculation.log")
        with mfa_open(log_path, "w", encoding="utf8") as log_file:
            # Reset all pronunciation counts to 0
            pron_df = self.polars_db.get_table("pronunciation")
            if not pron_df.is_empty():
                pron_df = pron_df.with_columns(pl.lit(0).alias("count"))
                self.polars_db.replace_table("pronunciation", pron_df)
            
            dictionaries = dict_df.to_dicts()
            dictionary_mappings = []
            for d in dictionaries:
                d_id = d["id"]
                if d_id not in dictionary_counters:
                    continue
                counter = dictionary_counters[d_id]
                log_file.write(f"For {d['name']}:\n")
                
                # Build the word list for this dictionary. Filter to only speech words or special words.
                words_df = self.polars_db.get_table("word").filter(
                    (pl.col("dictionary_id") == d_id) &
                    (
                        ((pl.col("word_type").is_in(WordType.speech_types())) & (pl.col("count") > 0)) |
                        (pl.col("word").is_in([d["cutoff_word"], d["oov_word"], d["laughter_word"], d["bracketed_word"]]))
                    )
                ).select("word")
                # Mimic tuples (w,) from the SQL query – used later in a loop.
                words = [(row["word"],) for row in words_df.to_dicts()]
                
                # Retrieve pronunciations by joining the word and pronunciation tables.
                word_tbl = self.polars_db.get_table("word").filter(pl.col("dictionary_id") == d_id)
                pron_tbl = self.polars_db.get_table("pronunciation")
                join_df = word_tbl.join(
                    pron_tbl,
                    left_on="id",
                    right_on="word_id",
                    how="inner",
                    suffix="_pron"
                )
                # Apply the same filter as for words.
                join_df = join_df.filter(
                    (((pl.col("word_type").is_in(WordType.speech_types())) & (pl.col("count") > 0)) |
                    (pl.col("word").is_in([d["cutoff_word"], d["oov_word"], d["laughter_word"], d["bracketed_word"]])))
                )
                # Select the needed columns.
                pronunciations_list = join_df.select([
                    "word",
                    "pronunciation",
                    pl.col("id_pron").alias("pron_id"),
                    "generated_by_rule"
                ]).to_dicts()
                # Filter pronunciations: if generated_by_rule then only keep if present in counter.
                pronunciations_filtered = [
                    (row["word"], row["pronunciation"], row["pron_id"])
                    for row in pronunciations_list
                    if (not row["generated_by_rule"] or row["pronunciation"] in
                        counter.word_pronunciation_counts.get(row["word"], {}))
                ]
                # Always append the cutoff model entry.
                pronunciations_filtered.append((d["cutoff_word"], "cutoff_model", None))
                
                pron_mapping = {}
                for w, p, p_id in pronunciations_filtered:
                    pron_mapping[(w, p)] = {"id": p_id}
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    # Increment count for smoothing.
                    counter.word_pronunciation_counts[w][p] += 1  
                
                for (w_tuple,) in words:
                    w = w_tuple
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    if w not in counter.word_pronunciation_counts:
                        continue
                    pron_counts = counter.word_pronunciation_counts[w]
                    max_value = max(pron_counts.values()) if pron_counts else 1
                    for p, c in pron_counts.items():
                        if self.position_dependent_phones:
                            p = re.sub(r"_[BSEI]\b", "", p)
                        if (w, p) in pron_mapping:
                            pron_mapping[(w, p)]["count"] = c
                            pron_mapping[(w, p)]["probability"] = format_probability(c / max_value)
                
                silence_count = sum(counter.silence_before_counts.values())
                non_silence_count = sum(counter.non_silence_before_counts.values())
                log_file.write(f"Total silence count was {silence_count}\n")
                log_file.write(f"Total non silence count was {non_silence_count}\n")
                silence_probability = (
                    format_probability(silence_count / (silence_count + non_silence_count))
                    if (silence_count + non_silence_count) > 0 else 0.0
                )
                silence_prob_sum += silence_probability
                silence_probabilities = {}
                for w, p, _ in pronunciations_filtered:
                    count = counter.silence_following_counts[(w, p)]
                    total_count = (
                        counter.silence_following_counts[(w, p)]
                        + counter.non_silence_following_counts[(w, p)]
                    )
                    pron_mapping[(w, p)]["silence_following_count"] = count
                    pron_mapping[(w, p)]["non_silence_following_count"] = counter.non_silence_following_counts[(w, p)]
                    w_p_silence_count = count + (silence_probability * lambda_2)
                    prob = (
                        format_probability(w_p_silence_count / (total_count + lambda_2))
                        if (total_count + lambda_2) > 0 else 0.0
                    )
                    silence_probabilities[(w, p)] = prob
                    if w not in {initial_key[0], final_key[0], self.silence_word}:
                        pron_mapping[(w, p)]["silence_after_probability"] = prob
                
                lambda_3 = 2
                bar_count_silence_wp = collections.defaultdict(float)
                bar_count_non_silence_wp = collections.defaultdict(float)
                for (w_p1, w_p2), counts in counter.ngram_counts.items():
                    silence_prob = silence_probabilities.get(w_p1, 0.01)
                    total = counts["silence"] + counts["non_silence"]
                    bar_count_silence_wp[w_p2] += total * silence_prob
                    bar_count_non_silence_wp[w_p2] += total * (1 - silence_prob)
                
                for w, p, _ in pronunciations_filtered:
                    if w in {initial_key[0], final_key[0], self.silence_word}:
                        continue
                    s_count = counter.silence_before_counts[(w, p)]
                    ns_count = counter.non_silence_before_counts[(w, p)]
                    pron_mapping[(w, p)]["silence_before_correction"] = format_correction(
                        (s_count + lambda_3) / (bar_count_silence_wp[(w, p)] + lambda_3)
                    )
                    pron_mapping[(w, p)]["non_silence_before_correction"] = format_correction(
                        (ns_count + lambda_3) / (bar_count_non_silence_wp[(w, p)] + lambda_3)
                    )
                
                # Update pronunciation rows via our in-memory bulk update.
                self.polars_db.bulk_update("pronunciation", list(pron_mapping.values()))
                
                # --- Process cutoff pronunciations ---
                df_word = self.polars_db.get_table("word")
                df_pron = self.polars_db.get_table("pronunciation")
                join_cutoff = df_pron.join(
                    df_word, left_on="word_id", right_on="id", how="inner", suffix="_word"
                )
                cutoff_df = join_cutoff.filter(
                    (pl.col("word") != d["cutoff_word"]) &
                    (pl.col("word").str.starts_with(d["cutoff_word"][:-1]))
                ).select(["id", "pronunciation"])
                cutoff_query = cutoff_df.to_dicts()
                cutoff_mappings = []
                # For the cutoff model, pop its mapping
                cutoff_model = pron_mapping.pop((d["cutoff_word"], "cutoff_model"))
                cutoff_not_model = pron_mapping.get((d["cutoff_word"], "spn"), {})
                for entry in cutoff_query:
                    data = dict(cutoff_not_model) if entry["pronunciation"] == d["cutoff_word"] else dict(cutoff_model)
                    data["id"] = entry["id"]
                    cutoff_mappings.append(data)
                if cutoff_mappings:
                    self.polars_db.bulk_update("pronunciation", cutoff_mappings)
                
                # Compute corrections for the initial and final silence markers.
                initial_silence_count = counter.silence_before_counts[initial_key] + (silence_probability * lambda_2)
                initial_non_silence_count = counter.non_silence_before_counts[initial_key] + ((1 - silence_probability) * lambda_2)
                initial_silence_probability = (
                    format_probability(initial_silence_count / (initial_silence_count + initial_non_silence_count))
                    if (initial_silence_count + initial_non_silence_count) > 0 else 0.0
                )
                final_silence_correction = format_correction(
                    (counter.silence_before_counts[final_key] + lambda_3) /
                    (bar_count_silence_wp[final_key] + lambda_3)
                )
                final_non_silence_correction = format_correction(
                    (counter.non_silence_before_counts[final_key] + lambda_3) /
                    (bar_count_non_silence_wp[final_key] + lambda_3)
                )
                initial_silence_prob_sum += initial_silence_probability
                final_silence_correction_sum += final_silence_correction
                final_non_silence_correction_sum += final_non_silence_correction
                
                dictionary_mappings.append({
                    "id": d_id,
                    "silence_probability": silence_probability,
                })
            
            # Update overall dictionary fields.
            num_dicts = self.num_dictionaries if self.num_dictionaries > 0 else 1
            self.silence_probability = format_probability(silence_prob_sum / num_dicts)
            self.initial_silence_probability = format_probability(initial_silence_prob_sum / num_dicts)
            self.final_silence_correction = format_probability(final_silence_correction_sum / num_dicts)
            self.final_non_silence_correction = final_non_silence_correction_sum / num_dicts
            
            # Update the Dictionary table.
            self.polars_db.bulk_update("dictionary", dictionary_mappings)
            
            # --- Process phonological rules ---
            rules_df = self.polars_db.get_table("phonological_rule")
            if not rules_df.is_empty():
                rules = rules_df.to_dicts()
                rules_for_deletion = []
                for r in rules:
                    base_count = 0
                    base_sil_after_count = 0
                    base_nonsil_after_count = 0
                    rule_count = 0
                    rule_sil_before_correction = 0
                    base_sil_before_correction = 0
                    rule_nonsil_before_correction = 0
                    base_nonsil_before_correction = 0
                    rule_sil_after_count = 0
                    rule_nonsil_after_count = 0
                    rule_correction_count = 0
                    base_correction_count = 0
                    # Use the rule’s regex (with the named groups removed) for filtering.
                    pattern = r["match_regex"].replace("?P<segment>", "").replace("?P<preceding>", "").replace("?P<following>", "")
                    df_non_app = self.polars_db.get_table("pronunciation").filter(
                        (pl.col("count") > 1) & (pl.col("pronunciation").str.contains(pattern))
                    )
                    for p in df_non_app.to_dicts():
                        base_count += p["count"]
                        if p.get("silence_before_correction"):
                            base_sil_before_correction += p["silence_before_correction"]
                            base_nonsil_before_correction += p["non_silence_before_correction"]
                            base_correction_count += 1
                        base_sil_after_count += p.get("silence_following_count") or 0
                        base_nonsil_after_count += p.get("non_silence_following_count") or 0
                    # Process associated rule pronunciations; we assume these are stored in r["pronunciations"].
                    for p in r.get("pronunciations", []):
                        rule_count += p.get("count", 0)
                        if p.get("silence_before_correction"):
                            rule_sil_before_correction += p["silence_before_correction"]
                            rule_nonsil_before_correction += p["non_silence_before_correction"]
                            rule_correction_count += 1
                        rule_sil_after_count += p.get("silence_following_count") or 0
                        rule_nonsil_after_count += p.get("non_silence_following_count") or 0
                    if not rule_count:
                        rules_for_deletion.append(r["id"])
                        continue
                    r["probability"] = (
                        format_probability(rule_count / (rule_count + base_count))
                        if (rule_count + base_count) > 0 else 0.0
                    )
                    if rule_correction_count:
                        rule_sil_before_correction /= rule_correction_count
                        rule_nonsil_before_correction /= rule_correction_count
                        if base_correction_count:
                            base_sil_before_correction /= base_correction_count
                            base_nonsil_before_correction /= base_correction_count
                        else:
                            base_sil_before_correction = 1.0
                            base_nonsil_before_correction = 1.0
                        r["silence_before_correction"] = format_correction(
                            rule_sil_before_correction - base_sil_before_correction, positive_only=False
                        )
                        r["non_silence_before_correction"] = format_correction(
                            rule_nonsil_before_correction - base_nonsil_before_correction, positive_only=False
                        )
                    silence_after_probability = (
                        format_probability(
                            (rule_sil_after_count + lambda_2) /
                            (rule_sil_after_count + rule_nonsil_after_count + lambda_2)
                        )
                        if (rule_sil_after_count + rule_nonsil_after_count + lambda_2) > 0 else 0.0
                    )
                    base_sil_after_probability = (
                        format_probability(
                            (base_sil_after_count + lambda_2) /
                            (base_sil_after_count + base_nonsil_after_count + lambda_2)
                        )
                        if (base_sil_after_count + base_nonsil_after_count + lambda_2) > 0 else 0.0
                    )
                    r["silence_after_probability"] = format_correction(
                        silence_after_probability / base_sil_after_probability
                    )
                # Delete rules marked for deletion and their rule applications
                if rules_for_deletion:
                    df_rule_app = self.polars_db.get_table("rule_applications").filter(~pl.col("rule_id").is_in(rules_for_deletion))
                    self.polars_db.replace_table("rule_applications", df_rule_app)
                    df_rules = self.polars_db.get_table("phonological_rule").filter(~pl.col("id").is_in(rules_for_deletion))
                    self.polars_db.replace_table("phonological_rule", df_rules)
                    # Also remove Pronunciation rows with count==0 and generated_by_rule==True.
                    df_pron = self.polars_db.get_table("pronunciation").filter(~((pl.col("count") == 0) & (pl.col("generated_by_rule") == True)))
                    self.polars_db.replace_table("pronunciation", df_pron)
                    # Log changes in pronunciation counts per dictionary.
                    join_df = self.polars_db.get_table("dictionary").join(
                        self.polars_db.get_table("word"), left_on="id", right_on="dictionary_id", how="inner"
                    ).join(
                        self.polars_db.get_table("pronunciation"), left_on="id", right_on="word_id", how="inner"
                    )
                    pronunciation_counts = join_df.groupby("name").agg(pl.count("id_right").alias("count"))
                    for row in pronunciation_counts.to_dicts():
                        logger.debug(f"{row['name']}: Reduced number of pronunciations to {row['count']}")
        logger.debug(f"Calculating pronunciation probabilities took {time.time() - begin:.3f} seconds")

    # def collect_alignments(self) -> None:
    #     """
    #     Process alignment archives to extract word or phone alignments

    #     See Also
    #     --------
    #     :class:`~montreal_forced_aligner.alignment.multiprocessing.AlignmentExtractionFunction`
    #         Multiprocessing function for extracting alignments
    #     :meth:`.CorpusAligner.alignment_extraction_arguments`
    #         Arguments for extraction
    #     """
    #     with self.session() as session:
    #         workflow = (
    #             session.query(CorpusWorkflow)
    #             .filter(CorpusWorkflow.current == True)  # noqa
    #             .first()
    #         )
    #         if workflow.alignments_collected:
    #             return
    #         if config.USE_POSTGRES:
    #             session.execute(sqlalchemy.text("ALTER TABLE word_interval DISABLE TRIGGER all"))
    #             session.execute(sqlalchemy.text("ALTER TABLE phone_interval DISABLE TRIGGER all"))
    #             session.commit()
    #         max_phone_interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
    #         if max_phone_interval_id is None:
    #             max_phone_interval_id = 0
    #         max_word_interval_id = session.query(sqlalchemy.func.max(WordInterval.id)).scalar()
    #         if max_word_interval_id is None:
    #             max_word_interval_id = 0
    #         mapping_id = session.query(sqlalchemy.func.max(Word.mapping_id)).scalar()
    #         if mapping_id is None:
    #             mapping_id = -1
    #         mapping_id += 1

    #         phone_to_phone_id = {}
    #         ds = session.query(Phone.id, Phone.mapping_id).all()
    #         for p_id, mapping_id in ds:
    #             phone_to_phone_id[mapping_id] = p_id
    #         pronunciation_mappings = {}
    #         word_mappings = {}
    #         dictionaries = session.query(Dictionary.id)
    #         for (dict_id,) in dictionaries:
    #             word_mappings[dict_id] = {}
    #             pronunciation_mappings[dict_id] = {}
    #             words = session.query(Word.word, Word.id).filter(
    #                 Word.dictionary_id == dict_id,
    #                 sqlalchemy.or_(Word.count > 0, Word.word.in_(self.specials_set)),
    #             )
    #             for w, w_id in words:
    #                 word_mappings[dict_id][w] = w_id
    #             pronunciations = (
    #                 session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
    #                 .join(Pronunciation.word)
    #                 .filter(
    #                     Word.dictionary_id == dict_id,
    #                     sqlalchemy.or_(Word.count > 0, Word.word.in_(self.specials_set)),
    #                 )
    #             )
    #             for w, pron, p_id in pronunciations:
    #                 pronunciation_mappings[dict_id][(w, pron)] = p_id
    #     word_index = self.get_next_primary_key(Word)

    #     logger.info(f"Collecting phone and word alignments from {workflow.name} lattices...")
    #     all_begin = time.time()
    #     arguments = self.alignment_extraction_arguments()
    #     has_words = False
    #     phone_interval_count = 0
    #     new_words = []
    #     if config.USE_POSTGRES:
    #         conn = self.db_engine.raw_connection()
    #         cursor = conn.cursor()
    #         word_buf = io.StringIO()
    #         phone_buf = io.StringIO()
    #     else:
    #         word_csv_path = self.working_directory.joinpath("word_intervals.csv")
    #         phone_csv_path = self.working_directory.joinpath("phone_intervals.csv")
    #         word_buf = open(word_csv_path, "w", encoding="utf8", newline="")
    #         phone_buf = open(phone_csv_path, "w", encoding="utf8", newline="")
    #     word_writer = csv.DictWriter(
    #         word_buf,
    #         [
    #             "id",
    #             "begin",
    #             "end",
    #             "utterance_id",
    #             "word_id",
    #             "pronunciation_id",
    #             "workflow_id",
    #         ],
    #     )
    #     phone_writer = csv.DictWriter(
    #         phone_buf,
    #         [
    #             "id",
    #             "begin",
    #             "end",
    #             "phone_goodness",
    #             "phone_id",
    #             "word_interval_id",
    #             "utterance_id",
    #             "workflow_id",
    #         ],
    #     )

    #     if not config.USE_POSTGRES:
    #         word_writer.writeheader()
    #         phone_writer.writeheader()
    #     for (
    #         utterance,
    #         dict_id,
    #         ctm,
    #     ) in run_kaldi_function(
    #         AlignmentExtractionFunction, arguments, total_count=self.num_current_utterances
    #     ):
    #         new_phone_interval_mappings = []
    #         new_word_interval_mappings = []
    #         for word_interval in ctm.word_intervals:
    #             if word_interval.label not in word_mappings[dict_id]:
    #                 new_words.append(
    #                     {
    #                         "id": word_index,
    #                         "mapping_id": mapping_id,
    #                         "word": word_interval.label,
    #                         "dictionary_id": 1,
    #                         "word_type": WordType.oov,
    #                     }
    #                 )
    #                 word_mappings[dict_id][word_interval.label] = word_index
    #                 word_id = word_index
    #                 word_index += 1
    #                 mapping_id += 1
    #             else:
    #                 word_id = word_mappings[dict_id][word_interval.label]
    #             max_word_interval_id += 1
    #             pronunciation_id = pronunciation_mappings[dict_id].get(
    #                 (word_interval.label, word_interval.pronunciation), None
    #             )

    #             new_word_interval_mappings.append(
    #                 {
    #                     "id": max_word_interval_id,
    #                     "begin": word_interval.begin,
    #                     "end": word_interval.end,
    #                     "word_id": word_id,
    #                     "pronunciation_id": pronunciation_id,
    #                     "utterance_id": utterance,
    #                     "workflow_id": workflow.id,
    #                 }
    #             )
    #             for interval in word_interval.phones:
    #                 max_phone_interval_id += 1
    #                 new_phone_interval_mappings.append(
    #                     {
    #                         "id": max_phone_interval_id,
    #                         "begin": interval.begin,
    #                         "end": interval.end,
    #                         "phone_id": phone_to_phone_id[interval.symbol],
    #                         "utterance_id": utterance,
    #                         "workflow_id": workflow.id,
    #                         "word_interval_id": max_word_interval_id,
    #                         "phone_goodness": interval.confidence if interval.confidence else 0.0,
    #                     }
    #                 )
    #         phone_writer.writerows(new_phone_interval_mappings)
    #         word_writer.writerows(new_word_interval_mappings)
    #         if new_word_interval_mappings:
    #             has_words = True
    #         if config.USE_POSTGRES and phone_interval_count > 100000:
    #             if has_words:
    #                 word_buf.seek(0)
    #                 cursor.copy_from(word_buf, WordInterval.__tablename__, sep=",", null="")
    #                 word_buf.truncate(0)
    #                 word_buf.seek(0)

    #             phone_buf.seek(0)
    #             cursor.copy_from(phone_buf, PhoneInterval.__tablename__, sep=",", null="")
    #             phone_buf.truncate(0)
    #             phone_buf.seek(0)
    #             conn.commit()
    #             cursor.close()
    #             conn.close()
    #             conn = self.db_engine.raw_connection()
    #             cursor = conn.cursor()

    #     if config.USE_POSTGRES:
    #         if word_buf.tell() != 0:
    #             word_buf.seek(0)
    #             cursor.copy_from(word_buf, WordInterval.__tablename__, sep=",", null="")
    #             word_buf.truncate(0)
    #             word_buf.seek(0)

    #         if phone_buf.tell() != 0:
    #             phone_buf.seek(0)
    #             cursor.copy_from(phone_buf, PhoneInterval.__tablename__, sep=",", null="")
    #             phone_buf.truncate(0)
    #             phone_buf.seek(0)
    #         conn.commit()
    #         cursor.close()
    #         conn.close()
    #     else:
    #         word_buf.close()
    #         phone_buf.close()
    #         if has_words:
    #             subprocess.check_call(
    #                 [
    #                     "sqlite3",
    #                     self.db_path.as_posix(),
    #                     "--cmd",
    #                     ".mode csv",
    #                     f".import {word_csv_path.as_posix()} word_interval_temp",
    #                 ]
    #             )
    #         subprocess.check_call(
    #             [
    #                 "sqlite3",
    #                 self.db_path.as_posix(),
    #                 "--cmd",
    #                 ".mode csv",
    #                 f".import {phone_csv_path.as_posix()} phone_interval_temp",
    #             ]
    #         )
    #     with self.session() as session:
    #         if new_words:
    #             session.execute(sqlalchemy.insert(Word).values(new_words))
    #             session.commit()

    #         if not config.USE_POSTGRES:
    #             session.execute(
    #                 sqlalchemy.text("INSERT INTO word_interval SELECT * from word_interval_temp")
    #             )
    #             session.execute(
    #                 sqlalchemy.text("INSERT INTO phone_interval SELECT * from phone_interval_temp")
    #             )
    #             session.commit()
    #             session.execute(sqlalchemy.text("DROP TABLE word_interval_temp"))
    #             session.execute(sqlalchemy.text("DROP TABLE phone_interval_temp"))
    #             session.commit()
    #         workflow = (
    #             session.query(CorpusWorkflow)
    #             .filter(CorpusWorkflow.current == True)  # noqa
    #             .first()
    #         )
    #         if (
    #             workflow.workflow_type is WorkflowType.transcription
    #             or workflow.workflow_type is WorkflowType.per_speaker_transcription
    #         ):
    #             query = (
    #                 session.query(Utterance)
    #                 .options(subqueryload(Utterance.word_intervals).joinedload(WordInterval.word))
    #                 .group_by(Utterance.id)
    #             )
    #             mapping = []
    #             for u in query:
    #                 text = [
    #                     x.word.word
    #                     for x in u.word_intervals
    #                     if x.word.word != self.silence_word and x.workflow_id == workflow.id
    #                 ]
    #                 mapping.append({"id": u.id, "transcription_text": " ".join(text)})
    #             bulk_update(session, Utterance, mapping)
    #         session.query(CorpusWorkflow).filter(CorpusWorkflow.current == True).update(  # noqa
    #             {CorpusWorkflow.alignments_collected: True}
    #         )
    #         session.commit()
    #     if config.USE_POSTGRES:
    #         conn = self.db_engine.connect()
    #         conn.execution_options(isolation_level="AUTOCOMMIT")
    #         conn.execute(sqlalchemy.text("VACUUM ANALYZE word_interval, phone_interval;"))
    #         conn.execute(sqlalchemy.text("ALTER TABLE word_interval ENABLE TRIGGER all"))
    #         conn.execute(sqlalchemy.text("ALTER TABLE phone_interval ENABLE TRIGGER all"))
    #         conn.close()
    #     logger.debug(f"Collecting alignments took {time.time() - all_begin:.3f} seconds")
    def collect_alignments(self) -> None:
        """
        Process alignment archives to extract word or phone alignments using the in‑memory Polars database.
        
        This method converts the original SQLAlchemy‐/file‑based operations into efficient in‑memory
        operations with Polars DataFrames. New rows for word intervals, phone intervals, and words are
        accumulated in lists and then appended to the corresponding tables.
        """

        # --- Get current workflow ---
        workflow_tbl = self.polars_db.get_table("corpus_workflow")
        current_workflow_df = workflow_tbl.filter(pl.col("current") == True)
        if current_workflow_df.is_empty():
            logger.error("No current workflow found.")
            return
        workflow = current_workflow_df.head(1).to_dicts()[0]
        if workflow.get("alignments_collected", False):
            return

        # --- Get current maximum IDs and mapping_id ---
        phone_interval_tbl = self.polars_db.get_table("phone_interval")
        max_phone_interval_id = int(phone_interval_tbl["id"].max()) if not phone_interval_tbl.is_empty() else 0

        word_interval_tbl = self.polars_db.get_table("word_interval")
        max_word_interval_id = int(word_interval_tbl["id"].max()) if not word_interval_tbl.is_empty() else 0

        word_tbl = self.polars_db.get_table("word")
        if "mapping_id" in word_tbl.columns and not word_tbl.is_empty():
            mapping_id = int(word_tbl["mapping_id"].max())
        else:
            mapping_id = -1
        mapping_id += 1

        # --- Build phone mapping ---
        phone_tbl = self.polars_db.get_table("phone")
        phone_to_phone_id = {}
        if not phone_tbl.is_empty():
            for row in phone_tbl.to_dicts():
                phone_to_phone_id[row["mapping_id"]] = row["id"]

        # --- Build dictionary mappings for words and pronunciations ---
        dict_df = self.polars_db.get_table("dictionary")
        word_mappings = {}
        pronunciation_mappings = {}
        for d in dict_df.to_dicts():
            dict_id = d["id"]
            word_mappings[dict_id] = {}
            pronunciation_mappings[dict_id] = {}

            # Get words from this dictionary that have a count > 0 or belong to specials.
            words_df = self.polars_db.get_table("word").filter(
                (pl.col("dictionary_id") == dict_id) &
                ((pl.col("count") > 0) | (pl.col("word").is_in(list(self.specials_set))))
            ).select(["word", "id"])
            for row in words_df.to_dicts():
                word_mappings[dict_id][row["word"]] = row["id"]

            # Join word and pronunciation tables to build pronunciation mappings.
            words_dict_df = self.polars_db.get_table("word").filter(pl.col("dictionary_id") == dict_id)
            pron_df = self.polars_db.get_table("pronunciation")
            join_df = words_dict_df.join(pron_df, left_on="id", right_on="word_id", how="inner", suffix="_pron")
            join_df = join_df.filter(
                (pl.col("count") > 0) | (pl.col("word").is_in(list(self.specials_set)))
            )
            # Expect the joined row to include "word", "pronunciation", and a unique pronunciation id under "id_pron".
            for row in join_df.select(["word", "pronunciation", "id_pron"]).to_dicts():
                pronunciation_mappings[dict_id][(row["word"], row["pronunciation"])] = row.get("id_pron")

        # --- Prepare for population of new rows ---
        word_index = self.polars_db.get_next_primary_key("word")
        logger.info(f"Collecting phone and word alignments from {workflow.get('name', 'unknown')} lattices...")
        all_begin = time.time()

        arguments = self.alignment_extraction_arguments()
        has_words = False
        new_words = []
        new_word_interval_rows = []
        new_phone_interval_rows = []

        # --- Process alignment extraction ---
        for utterance, dict_id, ctm in run_kaldi_function(
                AlignmentExtractionFunction, arguments, total_count=self.num_current_utterances
        ):
            for word_interval in ctm.word_intervals:
                # If the word is unseen, add it as OOV.
                if word_interval.label not in word_mappings[dict_id]:
                    new_words.append({
                        "id": word_index,
                        "mapping_id": mapping_id,
                        "word": word_interval.label,
                        "dictionary_id": 1,  # Adjust if needed; here we use 1 as in the original.
                        "word_type": WordType.oov,
                        "count": 0,
                    })
                    word_mappings[dict_id][word_interval.label] = word_index
                    word_id = word_index
                    word_index += 1
                    mapping_id += 1
                else:
                    word_id = word_mappings[dict_id][word_interval.label]
                max_word_interval_id += 1
                pronunciation_id = pronunciation_mappings[dict_id].get((word_interval.label, word_interval.pronunciation), None)
                new_word_interval_rows.append({
                    "id": max_word_interval_id,
                    "begin": word_interval.begin,
                    "end": word_interval.end,
                    "word_id": word_id,
                    "pronunciation_id": pronunciation_id,
                    "utterance_id": utterance,
                    "workflow_id": workflow["id"],
                })
                for interval in word_interval.phones:
                    max_phone_interval_id += 1
                    new_phone_interval_rows.append({
                        "id": max_phone_interval_id,
                        "begin": interval.begin,
                        "end": interval.end,
                        "phone_goodness": interval.confidence if interval.confidence else 0.0,
                        "phone_id": phone_to_phone_id.get(interval.symbol),
                        "utterance_id": utterance,
                        "workflow_id": workflow["id"],
                        "word_interval_id": max_word_interval_id,
                    })
            if new_word_interval_rows:
                has_words = True

        # --- Update in-memory tables ---
        if new_word_interval_rows:
            old_wi = self.polars_db.get_table("word_interval")
            new_wi_df = pl.DataFrame(new_word_interval_rows)
            self.polars_db.replace_table(
                "word_interval",
                pl.concat([old_wi, new_wi_df]) if not old_wi.is_empty() else new_wi_df
            )
        if new_phone_interval_rows:
            old_pi = self.polars_db.get_table("phone_interval")
            new_pi_df = pl.DataFrame(new_phone_interval_rows)
            self.polars_db.replace_table(
                "phone_interval",
                pl.concat([old_pi, new_pi_df]) if not old_pi.is_empty() else new_pi_df
            )
        if new_words:
            old_word_df = self.polars_db.get_table("word")
            new_word_df = pl.DataFrame(new_words)
            self.polars_db.replace_table(
                "word",
                pl.concat([old_word_df, new_word_df]) if not old_word_df.is_empty() else new_word_df
            )

        # --- Update transcriptions for utterances if needed ---
        if workflow.get("workflow_type") in (WorkflowType.transcription, WorkflowType.per_speaker_transcription):
            word_interval_df = self.polars_db.get_table("word_interval")
            word_df = self.polars_db.get_table("word")
            join_df = word_interval_df.join(word_df, left_on="word_id", right_on="id", how="left")
            join_df = join_df.filter(
                (pl.col("workflow_id") == workflow["id"]) & (pl.col("word") != self.silence_word)
            )
            join_df = join_df.sort("begin")
            utt_trans_df = join_df.groupby("utterance_id").agg(pl.col("word").implode())
            transcription_updates = []
            for row in utt_trans_df.to_dicts():
                transcription_updates.append({
                    "id": row["utterance_id"],
                    "transcription_text": " ".join(row["word"])
                })
            self.polars_db.bulk_update("utterance", transcription_updates)

        # --- Mark workflow as having collected alignments ---
        self.polars_db.bulk_update("corpus_workflow", [{"id": workflow["id"], "alignments_collected": True}])

        # Note: In this Polars-based implementation, there is no need to VACUUM or re-enable triggers.
        logger.debug(f"Collecting alignments took {time.time() - all_begin:.3f} seconds")
        
    # def fine_tune_alignments(self) -> None:
    #     """
    #     Fine tune aligned boundaries to millisecond precision
    #     """
    #     logger.info("Fine tuning alignments...")
    #     all_begin = time.time()
    #     with self.session() as session:
    #         arguments = self.fine_tune_arguments()
    #         update_mappings = []
    #         for result in run_kaldi_function(
    #             FineTuneFunction, arguments, total_count=self.num_utterances
    #         ):
    #             update_mappings.extend(result[0])
    #             update_mappings.extend([{"id": x, "begin": 0, "end": 0} for x in result[1]])
    #         bulk_update(session, PhoneInterval, update_mappings)
    #         session.flush()
    #         deleted_count = session.execute(
    #             PhoneInterval.__table__.delete().where(PhoneInterval.end == 0)
    #         )
    #         logger.debug(f"Deleted {deleted_count} phone intervals of zero duration")
    #         session.flush()
    #         word_update_mappings = []
    #         word_intervals = (
    #             session.query(
    #                 WordInterval.id,
    #                 sqlalchemy.func.min(PhoneInterval.begin),
    #                 sqlalchemy.func.max(PhoneInterval.end),
    #             )
    #             .join(PhoneInterval.word_interval)
    #             .group_by(WordInterval.id)
    #         )
    #         for wi_id, begin, end in word_intervals:
    #             word_update_mappings.append({"id": wi_id, "begin": begin, "end": end})
    #         bulk_update(session, WordInterval, word_update_mappings)
    #         session.commit()
    #         sq = (
    #             session.query(
    #                 WordInterval.id, sqlalchemy.func.count(PhoneInterval.id).label("phone_count")
    #             )
    #             .outerjoin(WordInterval.phone_intervals)
    #             .group_by(WordInterval.id)
    #         ).subquery()
    #         word_interval_deletions = session.query(sq.c.id).filter(sq.c.phone_count == 0)
    #         deleted_count = (
    #             session.query(WordInterval)
    #             .filter(WordInterval.id.in_(word_interval_deletions))
    #             .delete()
    #         )
    #         logger.debug(
    #             f"Deleted {deleted_count} word intervals no longer containing phone intervals"
    #         )
    #         session.commit()
    #     self.export_frame_shift = round(self.export_frame_shift / 10, 4)
    #     logger.debug(f"Fine tuning alignments took {time.time() - all_begin:.3f} seconds")
    def fine_tune_alignments(self) -> None:
        """
        Fine tune aligned boundaries to millisecond precision using in-memory Polars DataFrames.
        
        Instead of using SQLAlchemy sessions and raw SQL, this function updates the "phone_interval" 
        and "word_interval" tables stored in the PolarsDB and performs deletions/aggregations using 
        vectorized operations.
        """

        logger.info("Fine tuning alignments...")
        all_begin = time.time()

        # -------------------------------------------------------------------------
        # 1. Run fine tuning and update phone intervals.
        # -------------------------------------------------------------------------
        arguments = self.fine_tune_arguments()
        update_mappings = []

        # run_kaldi_function returns two lists per result.
        for result in run_kaldi_function(
            FineTuneFunction, arguments, total_count=self.num_utterances
        ):
            # Extend with updates; first list contains valid updates, second list errors
            update_mappings.extend(result[0])
            update_mappings.extend([{"id": x, "begin": 0, "end": 0} for x in result[1]])

        self.polars_db.bulk_update("phone_interval", update_mappings)

        # Delete any phone intervals that now have an 'end' value of zero.
        phone_interval_df = self.polars_db.get_table("phone_interval")
        old_count = phone_interval_df.height
        filtered_phone_df = phone_interval_df.filter(pl.col("end") != 0)
        deleted_phone_count = old_count - filtered_phone_df.height
        logger.debug(f"Deleted {deleted_phone_count} phone intervals of zero duration")
        self.polars_db.replace_table("phone_interval", filtered_phone_df)

        # -------------------------------------------------------------------------
        # 2. Update boundaries in word intervals based on the phone intervals.
        # -------------------------------------------------------------------------
        # Group phone intervals by their associated word_interval id to obtain new boundaries.
        if filtered_phone_df.is_empty():
            agg_df = pl.DataFrame([], schema=["word_interval_id", "min_begin", "max_end"])
        else:
            agg_df = filtered_phone_df.groupby("word_interval_id").agg([
                pl.col("begin").min().alias("min_begin"),
                pl.col("end").max().alias("max_end")
            ])
            
        word_update_mappings = [
            {"id": row["word_interval_id"], "begin": row["min_begin"], "end": row["max_end"]}
            for row in agg_df.to_dicts()
        ]
        self.polars_db.bulk_update("word_interval", word_update_mappings)

        # -------------------------------------------------------------------------
        # 3. Remove word intervals that no longer have any associated phone intervals.
        # -------------------------------------------------------------------------
        wi_df = self.polars_db.get_table("word_interval")
        # Identify word interval IDs that appear in phone intervals table.
        if filtered_phone_df.is_empty():
            # If there are no phone intervals at all, all word intervals should be deleted.
            word_intervals_to_delete = set(wi_df["id"].to_list() if not wi_df.is_empty() else [])
        else:
            wi_with_phone = set(filtered_phone_df["word_interval_id"].to_list())
            all_wi_ids = set(wi_df["id"].to_list())
            # Also, if a word interval does not appear at all in phone intervals, mark it for deletion.
            word_intervals_to_delete = all_wi_ids - wi_with_phone

            # For extra safety, if grouping by word_interval_id yields a count of 0, add those IDs.
            count_df = filtered_phone_df.groupby("word_interval_id").agg(pl.count("id").alias("phone_count"))
            for row in count_df.to_dicts():
                if row["phone_count"] == 0:
                    word_intervals_to_delete.add(row["word_interval_id"])
        
        # Filter out any word intervals whose id is in the deletion set.
        new_wi_df = wi_df.filter(~pl.col("id").is_in(list(word_intervals_to_delete)))
        deleted_wi_count = wi_df.height - new_wi_df.height
        logger.debug(f"Deleted {deleted_wi_count} word intervals no longer containing phone intervals")
        self.polars_db.replace_table("word_interval", new_wi_df)

        # -------------------------------------------------------------------------
        # 4. Adjust export frame shift value.
        # -------------------------------------------------------------------------
        self.export_frame_shift = round(self.export_frame_shift / 10, 4)
        logger.debug(f"Fine tuning alignments took {time.time() - all_begin:.3f} seconds")


    def fine_tune_arguments(self) -> List[FineTuneArguments]:
        """
        Generate Job arguments for :class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneFunction`

        Returns
        -------
        list[:class:`~montreal_forced_aligner.alignment.multiprocessing.FineTuneArguments`]
            Arguments for processing
        """
        args = []
        fst, group_table, phone_to_group_mapping = self.compile_phone_group_lexicon_fst()
        lexicon_compiler = LexiconCompiler(
            position_dependent_phones=self.position_dependent_phones,
            phones=self.non_silence_phones,
        )
        lexicon_compiler.word_table = group_table
        lexicon_compiler._fst = fst
        options = self.mfcc_options
        options["frame_shift"] = 1
        mfcc_computer = MfccComputer(**options)
        pitch_computer = None
        if self.use_pitch:
            options = self.pitch_options
            options["frame_shift"] = 1
            pitch_computer = PitchComputer(**options)
        align_options = self.align_options
        # align_options['transition_scale'] = align_options['transition_scale'] / 10
        align_options["acoustic_scale"] = 1.0
        for j in self.jobs:
            log_path = self.working_log_directory.joinpath(f"fine_tune.{j.id}.log")
            args.append(
                FineTuneArguments(
                    j.id,
                    getattr(self, "session" if config.USE_THREADING else "db_string", ""),
                    log_path,
                    mfcc_computer,
                    pitch_computer,
                    lexicon_compiler,
                    self.model_path,
                    self.tree_path,
                    align_options,
                    phone_to_group_mapping,
                    self.mfcc_computer.frame_shift,
                )
            )
        return args

    # def export_textgrids(
    #     self,
    #     output_format: str = TextFileType.TEXTGRID.value,
    #     include_original_text: bool = False,
    # ) -> None:
    #     """
    #     Exports alignments to TextGrid files

    #     See Also
    #     --------
    #     :class:`~montreal_forced_aligner.alignment.multiprocessing.ExportTextGridProcessWorker`
    #         Multiprocessing helper function for TextGrid export
    #     :meth:`.CorpusAligner.export_textgrid_arguments`
    #         Job method for TextGrid export

    #     Parameters
    #     ----------
    #     output_format: str, optional
    #         Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
    #     """
    #     workflow = self.current_workflow
    #     if not workflow.alignments_collected:
    #         self.collect_alignments()
    #     begin = time.time()
    #     error_dict = {}
    #     with tqdm(total=self.num_files, disable=config.QUIET) as pbar:
    #         if config.USE_MP and config.NUM_JOBS > 1:
    #             with self.session() as session:
    #                 files_per_job = math.ceil(self.num_files / len(self.jobs))
    #                 file_batches = [{}]
    #                 query = (
    #                     session.query(
    #                         File.id,
    #                         File.name,
    #                         File.relative_path,
    #                         SoundFile.duration,
    #                         TextFile.text_file_path,
    #                     )
    #                     .join(File.sound_file)
    #                     .join(File.text_file)
    #                 )
    #                 for file_id, file_name, relative_path, file_duration, text_file_path in query:
    #                     if len(file_batches[-1]) >= files_per_job:
    #                         file_batches.append({})
    #                     file_batches[-1][file_id] = (
    #                         file_name,
    #                         relative_path,
    #                         file_duration,
    #                         text_file_path,
    #                     )
    #             stopped = mp.Event()

    #             finished_adding = mp.Event()
    #             for_write_queue = mp.Queue()
    #             return_queue = mp.Queue()
    #             export_procs = []
    #             for j in range(config.NUM_JOBS):
    #                 export_proc = ExportTextGridProcessWorker(
    #                     self.db_string,
    #                     for_write_queue,
    #                     return_queue,
    #                     stopped,
    #                     finished_adding,
    #                     self.export_frame_shift,
    #                     config.CLEANUP_TEXTGRIDS,
    #                     self.clitic_marker,
    #                     self.export_output_directory,
    #                     output_format,
    #                     include_original_text,
    #                 )
    #                 export_proc.start()
    #                 export_procs.append(export_proc)
    #             try:
    #                 for batch in file_batches:
    #                     for_write_queue.put(batch)
    #                 time.sleep(1)
    #                 finished_adding.set()
    #                 while True:
    #                     try:
    #                         result = return_queue.get(timeout=1)
    #                         if isinstance(result, AlignmentExportError):
    #                             error_dict[getattr(result, "path", 0)] = result
    #                             continue
    #                         if self.stopped.is_set():
    #                             continue
    #                     except Empty:
    #                         for proc in export_procs:
    #                             if not proc.finished_processing.is_set():
    #                                 break
    #                         else:
    #                             break
    #                         continue
    #                     if isinstance(result, int):
    #                         pbar.update(result)
    #             except Exception:
    #                 stopped.set()
    #                 raise
    #             finally:
    #                 for p in export_procs:
    #                     p.join()
    #         else:
    #             logger.debug("Not using multiprocessing for TextGrid export")

    #             with self.session() as session:
    #                 file_batch = {}
    #                 query = (
    #                     session.query(
    #                         File.id,
    #                         File.name,
    #                         File.relative_path,
    #                         SoundFile.duration,
    #                         TextFile.text_file_path,
    #                     )
    #                     .join(File.sound_file)
    #                     .join(File.text_file)
    #                 )
    #                 for file_id, file_name, relative_path, file_duration, text_file_path in query:
    #                     file_batch[file_id] = (
    #                         file_name,
    #                         relative_path,
    #                         file_duration,
    #                         text_file_path,
    #                     )
    #             for _ in construct_textgrid_output(
    #                 session,
    #                 file_batch,
    #                 workflow,
    #                 config.CLEANUP_TEXTGRIDS,
    #                 self.clitic_marker,
    #                 self.export_output_directory,
    #                 self.export_frame_shift,
    #                 output_format,
    #                 include_original_text,
    #             ):
    #                 pbar.update(1)

    #     if error_dict:
    #         logger.warning(
    #             f"There were {len(error_dict)} errors encountered in generating TextGrids. "
    #             f"Check {os.path.join(self.export_output_directory, 'output_errors.txt')} "
    #             f"for more details"
    #         )
    #         output_textgrid_writing_errors(self.export_output_directory, error_dict)
    #         if config.DEBUG:
    #             for k, v in error_dict.items():
    #                 print(k)
    #                 raise v
    #     logger.info(f"Finished exporting TextGrids to {self.export_output_directory}!")
    #     logger.debug(f"Exported TextGrids in a total of {time.time() - begin:.3f} seconds")
    def export_textgrids(
        self,
        output_format: str = TextFileType.TEXTGRID.value,
        include_original_text: bool = False,
    ) -> None:
        """
        Exports alignments to TextGrid files using in-memory Polars tables.
        
        This function builds file information by joining the in‑memory "file", "sound_file", and
        "text_file" tables, and then dispatches the export jobs either via multiprocessing or in a
        single process.
        """
        # Ensure that alignments have been collected before exporting TextGrids.
        workflow = self.current_workflow
        if not workflow.alignments_collected:
            self.collect_alignments()
        
        begin = time.time()
        error_dict = {}
        
        with tqdm(total=self.num_files, disable=config.QUIET) as pbar:
            # Retrieve file details from the in‑memory Polars tables.
            df_file = self.polars_db.get_table("file")
            df_sound = self.polars_db.get_table("sound_file")
            df_text = self.polars_db.get_table("text_file")
            # Assumes a join key where sound_file and text_file have a "file_id" column.
            joined = (
                df_file.join(df_sound, left_on="id", right_on="file_id", how="left")
                    .join(df_text, left_on="id", right_on="file_id", how="left")
            )
            # Select only the required fields.
            file_info = joined.select([
                "id",
                "name",
                "relative_path",
                "duration",          # from sound_file
                "text_file_path"     # from text_file
            ]).to_dicts()
            
            if config.USE_MP and config.NUM_JOBS > 1:
                # Build file batches for multiprocessing.
                files_per_job = math.ceil(self.num_files / len(self.jobs))
                file_batches = [{}]
                for rec in file_info:
                    file_id = rec["id"]
                    file_entry = (rec["name"], rec["relative_path"], rec["duration"], rec["text_file_path"])
                    if len(file_batches[-1]) >= files_per_job:
                        file_batches.append({})
                    file_batches[-1][file_id] = file_entry
                
                # Set up multiprocessing events and queues.
                import multiprocessing as mp
                from queue import Empty
                stopped = mp.Event()
                finished_adding = mp.Event()
                for_write_queue = mp.Queue()
                return_queue = mp.Queue()
                export_procs = []
                # Start export processes.
                for j in range(config.NUM_JOBS):
                    export_proc = ExportTextGridProcessWorker(
                        self.db_string,
                        for_write_queue,
                        return_queue,
                        stopped,
                        finished_adding,
                        self.export_frame_shift,
                        config.CLEANUP_TEXTGRIDS,
                        self.clitic_marker,
                        self.export_output_directory,
                        output_format,
                        include_original_text,
                    )
                    export_proc.start()
                    export_procs.append(export_proc)
                try:
                    # Queue batches for processing.
                    for batch in file_batches:
                        for_write_queue.put(batch)
                    time.sleep(1)
                    finished_adding.set()
                    # Process results from worker processes.
                    while True:
                        try:
                            result = return_queue.get(timeout=1)
                            if isinstance(result, AlignmentExportError):
                                error_dict[getattr(result, "path", 0)] = result
                                continue
                            if stopped.is_set():
                                continue
                        except Empty:
                            # Check if all export processes have finished processing.
                            all_finished = True
                            for proc in export_procs:
                                if not proc.finished_processing.is_set():
                                    all_finished = False
                                    break
                            if all_finished:
                                break
                            continue
                        if isinstance(result, int):
                            pbar.update(result)
                except Exception:
                    stopped.set()
                    raise
                finally:
                    for p in export_procs:
                        p.join()
            else:
                logger.debug("Not using multiprocessing for TextGrid export")
                # Build a single file batch.
                file_batch = {}
                for rec in file_info:
                    file_batch[rec["id"]] = (
                        rec["name"],
                        rec["relative_path"],
                        rec["duration"],
                        rec["text_file_path"],
                    )
                # Call an updated construct_textgrid_output() that does not require a session.
                for _ in construct_textgrid_output(
                    file_batch,
                    workflow,
                    config.CLEANUP_TEXTGRIDS,
                    self.clitic_marker,
                    self.export_output_directory,
                    self.export_frame_shift,
                    output_format,
                    include_original_text,
                ):
                    pbar.update(1)
        
        if error_dict:
            logger.warning(
                f"There were {len(error_dict)} errors encountered in generating TextGrids. "
                f"Check {os.path.join(self.export_output_directory, 'output_errors.txt')} "
                f"for more details"
            )
            output_textgrid_writing_errors(self.export_output_directory, error_dict)
            if config.DEBUG:
                for k, v in error_dict.items():
                    print(k)
                    raise v
        logger.info(f"Finished exporting TextGrids to {self.export_output_directory}!")
        logger.debug(f"Exported TextGrids in a total of {time.time() - begin:.3f} seconds")


    def export_files(
        self,
        output_directory: typing.Union[Path, str],
        output_format: Optional[str] = None,
        include_original_text: bool = False,
    ) -> None:
        """
        Export a TextGrid file for every sound file in the dataset

        Parameters
        ----------
        output_directory: :class:`~pathlib.Path`
            Directory to save to
        output_format: str, optional
            Format to save alignments, one of 'long_textgrids' (the default), 'short_textgrids', or 'json', passed to praatio
        include_original_text: bool
            Flag for including the original text of the corpus files as a tier
        """
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)
        if output_format is None:
            output_format = TextFileType.TEXTGRID.value
        self.export_output_directory = output_directory

        logger.info(
            f"Exporting {self.current_workflow.name} TextGrids to {self.export_output_directory}..."
        )
        self.export_output_directory.mkdir(parents=True, exist_ok=True)
        analysis_csv = self.working_directory.joinpath("alignment_analysis.csv")
        if analysis_csv.exists():
            shutil.copyfile(
                analysis_csv, self.export_output_directory.joinpath("alignment_analysis.csv")
            )
        self.export_textgrids(output_format, include_original_text)

    # def validate_mapping(self, mapping: Dict[str, typing.Union[str, typing.List[str]]]):
    #     with self.session() as session:
    #         extra_phones = set(
    #             x[0]
    #             for x in session.query(Phone.phone).filter(Phone.phone_type == PhoneType.extra)
    #         )
    #         phones = set(
    #             x[0]
    #             for x in session.query(Phone.phone).filter(
    #                 Phone.phone_type == PhoneType.non_silence
    #             )
    #         )
    #         found_phones = set()
    #         found_extra_phones = set()
    #         for aligned_phones, ref_phones in mapping.items():
    #             aligned_phones = aligned_phones.split()
    #             if isinstance(ref_phones, str):
    #                 ref_phones = [ref_phones]
    #             found_phones.update(aligned_phones)
    #             found_extra_phones.update(ref_phones)
    #         unreferenced_phones = sorted(phones - found_phones)
    #         unreferenced_extra_phones = sorted(extra_phones - found_extra_phones)
    #         logger.debug(
    #             f"Phones not referenced in mapping file: {', '.join(unreferenced_phones)}"
    #         )
    #         logger.debug(
    #             f"Reference phones not referenced in mapping file: {', '.join(unreferenced_extra_phones)}"
    #         )
    def validate_mapping(self, mapping: Dict[str, Union[str, List[str]]]):
        """
        Validates the provided phone mapping against the phone and extra phone values
        stored in the in-memory Polars database.
        
        Parameters
        ----------
        mapping : dict
            A dictionary whose keys are strings containing space-separated aligned phones and
            values are either a single reference phone or a list of reference phones.
        """

        # Retrieve the phone table from the in-memory Polars database.
        phone_df = self.polars_db.get_table("phone")
        
        if phone_df.is_empty():
            logger.debug("Phone table is empty; nothing to validate.")
            return

        # Extract extra phones and non-silence phones in a vectorized manner.
        extra_phones = set(
            phone_df.filter(pl.col("phone_type") == PhoneType.extra)
                    .select("phone")
                    .to_series()
                    .to_list()
        )
        phones = set(
            phone_df.filter(pl.col("phone_type") == PhoneType.non_silence)
                    .select("phone")
                    .to_series()
                    .to_list()
        )

        found_phones = set()
        found_extra_phones = set()

        # Process the provided mapping.
        for aligned_phones, ref_phones in mapping.items():
            # Split the key string into individual phone tokens.
            aligned_list = aligned_phones.split()
            found_phones.update(aligned_list)
            if isinstance(ref_phones, str):
                ref_phones = [ref_phones]
            found_extra_phones.update(ref_phones)

        # Identify unreferenced phones.
        unreferenced_phones = sorted(phones - found_phones)
        unreferenced_extra_phones = sorted(extra_phones - found_extra_phones)

        logger.debug(
            f"Phones not referenced in mapping file: {', '.join(unreferenced_phones)}"
        )
        logger.debug(
            f"Reference phones not referenced in mapping file: {', '.join(unreferenced_extra_phones)}"
        )


    # def evaluate_alignments(
    #     self,
    #     mapping: Optional[Dict[str, str]] = None,
    #     output_directory: Optional[str] = None,
    #     comparison_source=WorkflowType.alignment,
    #     reference_source=WorkflowType.reference,
    # ) -> None:
    #     """
    #     Evaluate alignments against a reference directory

    #     Parameters
    #     ----------
    #     mapping: dict[str, Union[str, list[str]]], optional
    #         Mapping between phones that should be considered equal across different phone set types
    #     output_directory: str, optional
    #         Directory to save results, if not specified, it will be saved in the log directory
    #     comparison_source: :class:`~montreal_forced_aligner.data.WorkflowType`
    #         Workflow to compare to the reference intervals, defaults to :attr:`~montreal_forced_aligner.data.WorkflowType.alignment`
    #     reference_source: :class:`~montreal_forced_aligner.data.WorkflowType`
    #         Workflow to use as the reference intervals, defaults to :attr:`~montreal_forced_aligner.data.WorkflowType.reference`
    #     """

    #     all_begin = time.time()
    #     if output_directory:
    #         csv_path = os.path.join(
    #             output_directory,
    #             f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
    #         )
    #         confusion_path = os.path.join(
    #             output_directory,
    #             f"{comparison_source.name}_{reference_source.name}_confusions.csv",
    #         )
    #     else:
    #         self._current_workflow = "evaluation"
    #         os.makedirs(self.working_log_directory, exist_ok=True)
    #         csv_path = os.path.join(
    #             self.working_log_directory,
    #             f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
    #         )
    #         confusion_path = os.path.join(
    #             self.working_log_directory,
    #             f"{comparison_source.name}_{reference_source.name}_confusions.csv",
    #         )
    #     csv_header = [
    #         "file",
    #         "begin",
    #         "end",
    #         "speaker",
    #         "duration",
    #         "normalized_text",
    #         "oovs",
    #         "reference_phone_count",
    #         "alignment_score",
    #         "phone_error_rate",
    #         "alignment_log_likelihood",
    #         "word_count",
    #         "oov_count",
    #     ]

    #     score_count = 0
    #     score_sum = 0
    #     phone_edit_sum = 0
    #     phone_length_sum = 0
    #     phone_confusions = collections.Counter()
    #     with self.session() as session:
    #         # Set up
    #         logger.info("Evaluating alignments...")
    #         logger.debug(f"Mapping: {mapping}")
    #         reference_workflow_id = self.get_latest_workflow_run(reference_source, session).id
    #         comparison_workflow_id = self.get_latest_workflow_run(comparison_source, session).id
    #         update_mappings = []
    #         indices = []
    #         to_comp = []
    #         score_func = functools.partial(
    #             align_phones,
    #             silence_phone=self.optional_silence_phone,
    #             custom_mapping=mapping,
    #             debug=config.DEBUG,
    #         )
    #         unaligned_utts = []
    #         utterances: typing.List[Utterance] = session.query(Utterance).options(
    #             joinedload(Utterance.file, innerjoin=True),
    #             joinedload(Utterance.speaker, innerjoin=True),
    #             subqueryload(Utterance.phone_intervals).options(
    #                 joinedload(PhoneInterval.phone, innerjoin=True),
    #                 joinedload(PhoneInterval.workflow, innerjoin=True),
    #             ),
    #             subqueryload(Utterance.word_intervals).options(
    #                 joinedload(WordInterval.word, innerjoin=True),
    #                 joinedload(WordInterval.workflow, innerjoin=True),
    #             ),
    #         )
    #         reference_phone_counts = {}
    #         for u in utterances:
    #             reference_phones = u.phone_intervals_for_workflow(reference_workflow_id)
    #             comparison_phones = u.phone_intervals_for_workflow(comparison_workflow_id)
    #             if self.use_cutoff_model:
    #                 for wi in u.word_intervals:
    #                     if wi.workflow_id != comparison_workflow_id:
    #                         continue
    #                     if wi.word.word_type is WordType.cutoff:
    #                         comparison_phones = [
    #                             x
    #                             for x in comparison_phones
    #                             if x.end <= wi.begin or x.begin >= wi.end
    #                         ]
    #                         comparison_phones.append(
    #                             CtmInterval(begin=wi.begin, end=wi.end, label=self.oov_word)
    #                         )
    #                 comparison_phones = sorted(comparison_phones)

    #             reference_phone_counts[u.id] = len(reference_phones)
    #             if not reference_phone_counts[u.id]:
    #                 continue
    #             if not comparison_phones:  # couldn't be aligned
    #                 phone_error_rate = reference_phone_counts[u.id]
    #                 unaligned_utts.append(u)
    #                 update_mappings.append(
    #                     {
    #                         "id": u.id,
    #                         "alignment_score": None,
    #                         "phone_error_rate": phone_error_rate,
    #                     }
    #                 )
    #                 continue
    #             indices.append(u)
    #             to_comp.append((reference_phones, comparison_phones))
    #         with ThreadPool(config.NUM_JOBS) as pool:
    #             gen = pool.starmap(score_func, to_comp)
    #             for i, (score, phone_error_rate, errors) in enumerate(gen):
    #                 if score is None:
    #                     continue
    #                 u = indices[i]
    #                 phone_confusions.update(errors)
    #                 reference_phone_count = reference_phone_counts[u.id]
    #                 update_mappings.append(
    #                     {
    #                         "id": u.id,
    #                         "alignment_score": score,
    #                         "phone_error_rate": phone_error_rate,
    #                     }
    #                 )
    #                 score_count += 1
    #                 score_sum += score
    #                 phone_edit_sum += int(phone_error_rate * reference_phone_count)
    #                 phone_length_sum += reference_phone_count
    #         bulk_update(session, Utterance, update_mappings)
    #         self.alignment_evaluation_done = True
    #         session.query(Corpus).update({Corpus.alignment_evaluation_done: True})
    #         session.commit()
    #         logger.info("Exporting evaluation...")
    #         with mfa_open(csv_path, "w") as f:
    #             writer = csv.DictWriter(f, fieldnames=csv_header)
    #             writer.writeheader()
    #             utterances = (
    #                 session.query(
    #                     Utterance.id,
    #                     File.name,
    #                     Utterance.begin,
    #                     Utterance.end,
    #                     Speaker.name,
    #                     Utterance.duration,
    #                     Utterance.normalized_text,
    #                     Utterance.oovs,
    #                     Utterance.alignment_score,
    #                     Utterance.phone_error_rate,
    #                     Utterance.alignment_log_likelihood,
    #                 )
    #                 .join(Utterance.speaker)
    #                 .join(Utterance.file)
    #             )
    #             for (
    #                 u_id,
    #                 file_name,
    #                 begin,
    #                 end,
    #                 speaker_name,
    #                 duration,
    #                 normalized_text,
    #                 oovs,
    #                 alignment_score,
    #                 phone_error_rate,
    #                 alignment_log_likelihood,
    #             ) in utterances:
    #                 data = {
    #                     "file": file_name,
    #                     "begin": begin,
    #                     "end": end,
    #                     "duration": duration,
    #                     "speaker": speaker_name,
    #                     "normalized_text": normalized_text,
    #                     "oovs": oovs,
    #                     "reference_phone_count": reference_phone_counts[u_id],
    #                     "alignment_score": alignment_score,
    #                     "phone_error_rate": phone_error_rate,
    #                     "alignment_log_likelihood": alignment_log_likelihood,
    #                 }
    #                 data["word_count"] = len(data["normalized_text"].split())
    #                 data["oov_count"] = len(data["oovs"].split())
    #                 if alignment_score is not None:
    #                     score_count += 1
    #                     score_sum += alignment_score
    #                 writer.writerow(data)
    #     with mfa_open(confusion_path, "w") as f:
    #         f.write("reference,hypothesis,count\n")
    #         for k, v in sorted(phone_confusions.items(), key=lambda x: -x[1]):
    #             f.write(f"{k[0]},{k[1]},{v}\n")
    #     logger.info(f"Average overlap score: {score_sum/score_count}")
    #     logger.info(f"Average phone error rate: {phone_edit_sum/phone_length_sum}")
    #     logger.debug(f"Alignment evaluation took {time.time()-all_begin} seconds")
    def evaluate_alignments(
        self,
        mapping: Optional[Dict[str, str]] = None,
        output_directory: Optional[str] = None,
        comparison_source=WorkflowType.alignment,
        reference_source=WorkflowType.reference,
    ) -> None:
        """
        Evaluate alignments against a reference by comparing phone intervals from two workflow runs.
        The evaluation results (and phone confusions) are exported to CSV files.
        
        Parameters
        ----------
        mapping : dict[str, Union[str, list[str]]], optional
            Mapping between phones that should be considered equal across different phone set types.
        output_directory : str, optional
            Directory to save results. If not specified, results will be saved in the working log directory.
        comparison_source : WorkflowType
            Workflow to compare to the reference intervals (default: alignment).
        reference_source : WorkflowType
            Workflow to use as the reference intervals (default: reference).
        """
        
        all_begin = time.time()
        if output_directory:
            csv_path = os.path.join(
                output_directory,
                f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
            )
            confusion_path = os.path.join(
                output_directory,
                f"{comparison_source.name}_{reference_source.name}_confusions.csv",
            )
        else:
            self._current_workflow = "evaluation"
            os.makedirs(self.working_log_directory, exist_ok=True)
            csv_path = os.path.join(
                self.working_log_directory,
                f"{comparison_source.name}_{reference_source.name}_evaluation.csv",
            )
            confusion_path = os.path.join(
                self.working_log_directory,
                f"{comparison_source.name}_{reference_source.name}_confusions.csv",
            )
        csv_header = [
            "file",
            "begin",
            "end",
            "speaker",
            "duration",
            "normalized_text",
            "oovs",
            "reference_phone_count",
            "alignment_score",
            "phone_error_rate",
            "alignment_log_likelihood",
            "word_count",
            "oov_count",
        ]

        score_count = 0
        score_sum = 0
        phone_edit_sum = 0
        phone_length_sum = 0
        phone_confusions = collections.Counter()

        # Get the latest workflow runs using adapted methods (now operating on in-memory tables)
        reference_workflow = self.get_latest_workflow_run(reference_source)
        comparison_workflow = self.get_latest_workflow_run(comparison_source)
        reference_workflow_id = reference_workflow["id"]
        comparison_workflow_id = comparison_workflow["id"]

        update_mappings = []
        indices = []
        to_comp = []
        unaligned_utts = []
        score_func = functools.partial(
            align_phones,
            silence_phone=self.optional_silence_phone,
            custom_mapping=mapping,
            debug=config.DEBUG,
        )

        # Retrieve tables from the Polars-based database
        utterances = self.polars_db.get_table("utterance").to_dicts()
        phone_df = self.polars_db.get_table("phone_interval")
        word_interval_df = self.polars_db.get_table("word_interval")
        # Build a mapping of word_id to word info (to check for cutoff words)
        word_map = {w["id"]: w for w in self.polars_db.get_table("word").to_dicts()}

        reference_phone_counts = {}
        # For each utterance, compute the reference and comparison phone intervals.
        for u in utterances:
            u_id = u["id"]
            ref_phones = phone_df.filter(
                (pl.col("utterance_id") == u_id) & (pl.col("workflow_id") == reference_workflow_id)
            ).to_dicts()
            comp_phones = phone_df.filter(
                (pl.col("utterance_id") == u_id) & (pl.col("workflow_id") == comparison_workflow_id)
            ).to_dicts()
            # If using cutoff models, remove phone intervals overlapping words marked as cutoff and
            # substitute the cutoff interval.
            if self.use_cutoff_model:
                wi_list = word_interval_df.filter(pl.col("utterance_id") == u_id).to_dicts()
                for wi in wi_list:
                    if wi["workflow_id"] != comparison_workflow_id:
                        continue
                    word_info = word_map.get(wi["word_id"], {})
                    if word_info.get("word_type") == WordType.cutoff:
                        # Remove overlapping phone intervals.
                        comp_phones = [
                            p
                            for p in comp_phones
                            if (p.get("end", 0) <= wi["begin"] or p.get("begin", 0) >= wi["end"])
                        ]
                        # Append the cutoff interval as a CtmInterval (or equivalent dict)
                        comp_phones.append({
                            "begin": wi["begin"],
                            "end": wi["end"],
                            "label": self.oov_word,
                        })
                # Sort the comparison phones by onset.
                comp_phones = sorted(comp_phones, key=lambda x: x.get("begin", 0))
            reference_phone_counts[u_id] = len(ref_phones)
            if reference_phone_counts[u_id] == 0:
                continue
            if not comp_phones:  # If no comparison phones found, treat as unaligned.
                phone_error_rate = reference_phone_counts[u_id]
                unaligned_utts.append(u)
                update_mappings.append({
                    "id": u_id,
                    "alignment_score": None,
                    "phone_error_rate": phone_error_rate,
                })
                continue
            indices.append(u)
            to_comp.append((ref_phones, comp_phones))

        # Score the alignments concurrently using a thread pool
        if to_comp:
            with ThreadPool(config.NUM_JOBS) as pool:
                results = pool.starmap(score_func, to_comp)
                for i, (score, phone_error_rate, errors) in enumerate(results):
                    if score is None:
                        continue
                    u = indices[i]
                    phone_confusions.update(errors)
                    ref_count = reference_phone_counts[u["id"]]
                    update_mappings.append({
                        "id": u["id"],
                        "alignment_score": score,
                        "phone_error_rate": phone_error_rate,
                    })
                    score_count += 1
                    score_sum += score
                    phone_edit_sum += int(phone_error_rate * ref_count)
                    phone_length_sum += ref_count

        # Update the utterance table with evaluation scores.
        self.polars_db.bulk_update("utterance", update_mappings)
        self.alignment_evaluation_done = True
        # Mark the corpus as having finished alignment evaluation.
        corpus_df = self.polars_db.get_table("corpus")
        corpus_df = corpus_df.with_columns(pl.lit(True).alias("alignment_evaluation_done"))
        self.polars_db.replace_table("corpus", corpus_df)

        logger.info("Exporting evaluation...")
        # Export evaluation results by joining utterance, file, and speaker tables.
        utt_df = self.polars_db.get_table("utterance")
        file_df = self.polars_db.get_table("file")
        speaker_df = self.polars_db.get_table("speaker")
        join_df = utt_df.join(file_df, left_on="file_id", right_on="id", how="left") \
                        .join(speaker_df, left_on="speaker_id", right_on="id", how="left", suffix="_spk")
        # Select necessary columns.
        eval_df = join_df.select([
            pl.col("id").alias("utt_id"),
            pl.col("name"),
            pl.col("begin"),
            pl.col("end"),
            pl.col("name_spk").alias("speaker"),
            pl.col("duration"),
            pl.col("normalized_text"),
            pl.col("oovs"),
            pl.col("alignment_score"),
            pl.col("phone_error_rate"),
            pl.col("alignment_log_likelihood"),
        ])

        # Convert evaluation DataFrame to a list of dictionaries and add computed fields.
        eval_list = []
        for row in eval_df.to_dicts():
            u_id = row["utt_id"]
            row["reference_phone_count"] = reference_phone_counts.get(u_id, 0)
            row["word_count"] = len(row["normalized_text"].split()) if row.get("normalized_text") else 0
            row["oov_count"] = len(row["oovs"].split()) if row.get("oovs") else 0
            eval_list.append(row)

        # Write evaluation results CSV.
        with mfa_open(csv_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            for row in eval_list:
                writer.writerow(row)

        # Write phone confusion information.
        with mfa_open(confusion_path, "w") as f:
            f.write("reference,hypothesis,count\n")
            for (ref_phone, hyp_phone), cnt in sorted(phone_confusions.items(), key=lambda x: -x[1]):
                f.write(f"{ref_phone},{hyp_phone},{cnt}\n")

        logger.info(f"Average overlap score: {score_sum / score_count if score_count else 0}")
        logger.info(f"Average phone error rate: {phone_edit_sum / phone_length_sum if phone_length_sum else 0}")
        logger.debug(f"Alignment evaluation took {time.time() - all_begin} seconds")
