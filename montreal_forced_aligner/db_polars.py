"""
Polars-based database interface for the Montreal Forced Aligner.

This module replaces time‑consuming SQLAlchemy queries by loading key tables
(e.g., word, pronunciation, phone, dictionary, speaker, file, utterance, etc.)
into Polars DataFrames and providing helper methods for in‑memory updates.
No data is saved persistently; all tables are initialized as empty.
"""

import librosa
from pathlib import Path
import polars as pl
import numpy as np
import re, typing, datetime, os
from kalpy.data import KaldiMapping, Segment
from kalpy.feat.data import FeatureArchive
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utterance import Utterance as KalpyUtterance
from praatio import textgrid
from praatio.utilities.constants import Interval
from montreal_forced_aligner.data import (
    CtmInterval,
    PhoneSetType,
    PhoneType,
    TextFileType,
    WordType,
    WorkflowType,
)
from montreal_forced_aligner.helper import mfa_open

class PolarsDB:
    def __init__(self):
        """
        Initialize the PolarsDB with empty tables and predefined columns according
        to the corresponding PolarsModel classes. No data will be saved persistently;
        each table is an in-memory Polars DataFrame with an explicitly defined schema.
        """

        # Define the table schemas based on the columns from the PolarsModel subclasses.
        # (Only scalar columns are predefined; relationships/internal attributes are omitted.)
        table_schemas = {
            "dictionary_job": {
                "id": pl.Int64,
                "dictionary_id": pl.Int64,
                "job_id": pl.Int64,
            },
            "speaker_ordering": {
                "id": pl.Int64,
                "speaker_id": pl.Int64,
                "file_id": pl.Int64,
                "index": pl.Int64,
            },
            "corpus": {
                "id": pl.Int64,
                "name": pl.Utf8,
                "path": pl.Utf8,
                "data_directory": pl.Utf8,
                "imported": pl.Boolean,
                "text_normalized": pl.Boolean,
                "cutoffs_found": pl.Boolean,
                "features_generated": pl.Boolean,
                "vad_calculated": pl.Boolean,
                "ivectors_calculated": pl.Boolean,
                "plda_calculated": pl.Boolean,
                "xvectors_loaded": pl.Boolean,
                "alignment_done": pl.Boolean,
                "transcription_done": pl.Boolean,
                "alignment_evaluation_done": pl.Boolean,
                "has_reference_alignments": pl.Boolean,
                "has_sound_files": pl.Boolean,
                "has_text_files": pl.Boolean,
                "num_jobs": pl.Int64,
                "current_subset": pl.Int64,
            },
            "dialect": {
                "id": pl.Int64,
                "name": pl.Utf8,
            },
            "dictionary": {
                "id": pl.Int64,
                "name": pl.Utf8,
                "path": pl.Utf8,
                "rules_applied": pl.Boolean,
                "phone_set_type": pl.Utf8,
                "root_temp_directory": pl.Utf8,
                "clitic_cleanup_regex": pl.Utf8,
                "bracket_regex": pl.Utf8,
                "laughter_regex": pl.Utf8,
                "position_dependent_phones": pl.Boolean,
                "default": pl.Boolean,
                "clitic_marker": pl.Utf8,
                "silence_word": pl.Utf8,
                "optional_silence_phone": pl.Utf8,
                "oov_word": pl.Utf8,
                "oov_phone": pl.Utf8,
                "bracketed_word": pl.Utf8,
                "cutoff_word": pl.Utf8,
                "laughter_word": pl.Utf8,
                "use_g2p": pl.Boolean,
                "max_disambiguation_symbol": pl.Int64,
                "silence_probability": pl.Float64,
                "initial_silence_probability": pl.Float64,
                "final_silence_correction": pl.Float64,
                "final_non_silence_correction": pl.Float64,
                "dialect_id": pl.Int64,
            },
            "word": {
                "id": pl.Int64,
                "mapping_id": pl.Int64,
                "word": pl.Utf8,
                "count": pl.Int64,
                "word_type": pl.Object,
                "included": pl.Boolean,
                "initial_cost": pl.Float64,
                "final_cost": pl.Float64,
                "dictionary_id": pl.Int64,
            },
            "pronunciation": {
                "id": pl.Int64,
                "pronunciation": pl.Utf8,
                "probability": pl.Float64,
                "disambiguation": pl.Int64,
                "silence_after_probability": pl.Float64,
                "silence_before_correction": pl.Float64,
                "non_silence_before_correction": pl.Float64,
                "generated_by_rule": pl.Boolean,
                "count": pl.Int64,
                "silence_following_count": pl.Int64,
                "non_silence_following_count": pl.Int64,
                "word_id": pl.Int64,
                "word": pl.Object,
                "rules": pl.Object,
                "word_intervals": pl.Object,
            },
            "phonological_rule": {
                "id": pl.Int64,
                "segment": pl.Utf8,
                "preceding_context": pl.Utf8,
                "following_context": pl.Utf8,
                "replacement": pl.Utf8,
                "probability": pl.Float64,
                "silence_after_probability": pl.Float64,
                "silence_before_correction": pl.Float64,
                "non_silence_before_correction": pl.Float64,
                "dialect_id": pl.Int64,
                "dialect": pl.Object,
            },
            "rule_applications": {
                "pronunciation_id": pl.Int64,
                "rule_id": pl.Int64,
                "pronunciation": pl.Object,
                "rule": pl.Object,
            },
            "speaker": {
                "id": pl.Int64,
                "name": pl.Utf8,
                "cmvn": pl.Utf8,
                "fmllr": pl.Utf8,
                "min_f0": pl.Float64,
                "max_f0": pl.Float64,
                "ivector": pl.Object,
                "plda_vector": pl.Object,
                "xvector": pl.Object,
                "num_utterances": pl.Int64,
                "modified": pl.Boolean,
                "dictionary_id": pl.Int64,
            },
            "grapheme": {
                "id": pl.Int64,
                "mapping_id": pl.Int64,
                "grapheme": pl.Utf8,
            },
            "phone": {
                "id": pl.Int64,
                "mapping_id": pl.Int64,
                "phone": pl.Utf8,
                "kaldi_label": pl.Utf8,
                "position": pl.Utf8,
                "phone_type": pl.Int64,
                "mean_duration": pl.Float64,
                "sd_duration": pl.Float64,
            },
            "file": {
                "id": pl.Int64,
                "name": pl.Utf8,
                "relative_path": pl.Utf8,
                "modified": pl.Boolean,
                "speakers": pl.Object,   
                "text_file": pl.Object,  
                "sound_file": pl.Object, 
                "utterances": pl.Object,
            },
            "sound_file": {
                "file_id": pl.Int64,
                "file": pl.Object,
                "sound_file_path": pl.Utf8,
                "format": pl.Utf8,
                "sample_rate": pl.Int64,
                "duration": pl.Float64,
                "num_channels": pl.Int64,
                "sox_string": pl.Utf8,
            },
            "text_file": {
                "file_id": pl.Int64,
                "text_file_path": pl.Utf8,
                "file_type": pl.Utf8,
                "file": pl.Object,
            },
            
            "utterance": {
                "id": pl.Int64,
                "begin": pl.Float64,
                "end": pl.Float64,
                "channel": pl.Int64,
                "num_frames": pl.Int64,
                "text": pl.Utf8,
                "oovs": pl.Utf8,
                "normalized_text": pl.Utf8,
                "normalized_character_text": pl.Utf8,
                "transcription_text": pl.Utf8,
                "features": pl.Utf8,
                "ivector_ark": pl.Utf8,
                "vad_ark": pl.Utf8,
                "in_subset": pl.Boolean,
                "ignored": pl.Boolean,
                "alignment_log_likelihood": pl.Float64,
                "speech_log_likelihood": pl.Float64,
                "duration_deviation": pl.Float64,
                "phone_error_rate": pl.Float64,
                "alignment_score": pl.Float64,
                "word_error_rate": pl.Float64,
                "character_error_rate": pl.Float64,
                "ivector": pl.Utf8,
                "plda_vector": pl.Utf8,
                "xvector": pl.Utf8,
                "file_id": pl.Int64,
                "speaker_id": pl.Int64,
                "job_id": pl.Int64,
            },
            "corpus_workflow": {
                "id": pl.Int64,
                "name": pl.Utf8,
                "workflow_type": pl.Utf8,
                "working_directory": pl.Utf8,
                "time_stamp": pl.Datetime,
                "current": pl.Boolean,
                "done": pl.Boolean,
                "dirty": pl.Boolean,
                "alignments_collected": pl.Boolean,
                "score": pl.Float64,
            },
            "phone_interval": {
                "id": pl.Int64,
                "begin": pl.Float64, 
                "end": pl.Float64,
                "phone_goodness": pl.Float64,
                "label": pl.Utf8,
                "phone_id": pl.Int64,
                "phone": pl.Object,
                "word_interval_id": pl.Int64,
                "word_interval": pl.Object,
                "utterance_id": pl.Int64,
                "utterance": pl.Object,
                "workflow_id": pl.Int64,
                "workflow": pl.Object,
            },
            "word_interval": {
                "id": pl.Int64,
                "begin": pl.Float64,
                "end": pl.Float64,
                "label": pl.Utf8,
                "utterance_id": pl.Int64,
                "utterance": pl.Object,
                "word_id": pl.Int64,
                "word": pl.Object,
                "pronunciation_id": pl.Int64,
                "pronunciation": pl.Object,
                "workflow_id": pl.Int64,
                "workflow": pl.Object,
                "phone_intervals": pl.Object,
            },
            "job": {
                "id": pl.Int64,
                "corpus_id": pl.Int64,
                "corpus": pl.Object,
                "utterances": pl.Object,
                "symbols": pl.Object,
                "words": pl.Object,
                "dictionaries": pl.Object,
            },
            "m2m_symbol": {
                "id": pl.Int64,
                "symbol": pl.Utf8,
                "total_order": pl.Int64,
                "max_order": pl.Int64,
                "grapheme_order": pl.Int64,
                "phone_order": pl.Int64,
                "weight": pl.Float64,
                "jobs": pl.Object,
            },
            "m2m_job": {
                "m2m_id": pl.Int64,
                "job_id": pl.Int64,
                "m2m_symbol": pl.Object,
                "job": pl.Object,
            },
            "word_job": {
                "word_id": pl.Int64,
                "job_id": pl.Int64,
                "training": pl.Boolean,
                "word": pl.Object,
                "job": pl.Object,
            },
        }

        self.tables = {}
        for table_name, schema in table_schemas.items():
            # Use an empty list for each column
            empty_data = {col: [] for col in schema.keys()}
            # Create DataFrame and immediately cast columns to the schema-specified types
            self.tables[table_name] = (
                pl.DataFrame(empty_data)
                .with_columns([
                    pl.col(col).cast(dtype) for col, dtype in schema.items()
                ])
            )
        

    def get_table(self, table_name: str) -> pl.DataFrame:
        """
        Retrieve the DataFrame for a given table.

        Parameters
        ----------
        table_name : str
            The name of the table (e.g., "word", "speaker", etc).

        Returns
        -------
        pl.DataFrame
            The corresponding in-memory DataFrame.
        """
        if table_name not in self.tables:
            # Initialize an empty table if it does not exist.
            self.tables[table_name] = pl.DataFrame()
        return self.tables[table_name]

    def add_row(self, table_name: str, row: dict) -> None:
        """
        Append a row to the specified table in memory.

        Parameters
        ----------
        table_name : str
            Name of the table to update.
        row : dict
            Dictionary containing column names and values for the new row.
        """
        df = self.get_table(table_name)
        new_row_df = pl.DataFrame([row])
        if df.is_empty():
            self.tables[table_name] = new_row_df
        else:
            self.tables[table_name] = pl.concat([df, new_row_df])


    def add_rows(self, table_name: str, rows: list[dict]) -> None:
        """
        Add multiple rows to the specified table using a vectorized concatenation.
        
        Parameters
        ----------
        table_name : str
            The name of the table to update.
        rows : list of dict
            A list of dictionaries where each dictionary represents a row to add.
        """
        new_rows_df = pl.DataFrame(rows)
        self.tables[table_name] = pl.concat([self.tables[table_name], new_rows_df], how="diagonal")

    def update_row(self, table_name: str, update_id: int, update_mapping: dict) -> None:
        """
        Update a single row in the specified table. For the row where the "id" matches
        update_id, update the provided columns with the new values.

        Parameters
        ----------
        table_name : str
            The name of the table to update.
        update_id : int
            The primary key value (expected from the "id" column) of the row to update.
        update_mapping : dict
            A dictionary where keys are column names and values are the new values to set.
        """
        df = self.get_table(table_name)
        for key, value in update_mapping.items():
            # Skip updating the primary key if it is included
            if key == "id":
                continue

            if key in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("id") == update_id)
                    .then(value)
                    .otherwise(pl.col(key))
                    .alias(key)
                )
            else:
                # If the column doesn't exist, create it then update the target row
                df = df.with_columns(pl.lit(None).alias(key))
                df = df.with_columns(
                    pl.when(pl.col("id") == update_id)
                    .then(value)
                    .otherwise(pl.col(key))
                    .alias(key)
                )
        self.tables[table_name] = df

    def bulk_update(self, table_name: str, update_mappings: list[dict], id_field: str = "id") -> None:
        """
        Perform a bulk update on a table. For each mapping (expected to contain an id field specified by id_field
        along with columns to update), the row(s) with the matching id_field will have the specified columns updated.

        Parameters
        ----------
        table_name : str
            The name of the table to update.
        update_mappings : list of dict
            A list of dictionaries specifying the updates.
            Each dictionary must contain an identifier for the row to update using the key provided by id_field.
        id_field : str, optional
            The name of the identifier column (default is "id").
        """
        df = self.get_table(table_name)
        for mapping in update_mappings:
            update_id = mapping.get(id_field)
            # For each field (other than id_field), update the column conditionally.
            for key, value in mapping.items():
                if key == id_field:
                    continue
                # Only update if the column exists. Otherwise, add it.
                if key in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(id_field) == update_id)
                        .then(value)
                        .otherwise(pl.col(key))
                        .alias(key)
                    )
                else:
                    # Add a new column and then update the target row.
                    df = df.with_columns(pl.lit(None).alias(key))
                    df = df.with_columns(
                        pl.when(pl.col(id_field) == update_id)
                        .then(value)
                        .otherwise(pl.col(key))
                        .alias(key)
                    )
        self.tables[table_name] = df

    def replace_table(self, table_name: str, new_dataframe: pl.DataFrame) -> None:
        """
        Replace the in-memory table with a new DataFrame.

        Parameters
        ----------
        table_name : str
            The name of the table to replace.
        new_dataframe : pl.DataFrame
            The new DataFrame replacing the previous table.
        """
        self.tables[table_name] = new_dataframe

    def update_table(self, table_name: str, new_dataframe: pl.DataFrame) -> None:
        """
        Update a table with a new DataFrame.

        This method is provided for compatibility with callers expecting an 'update_table'
        method. It functions as an alias for replace_table.
        """
        self.replace_table(table_name, new_dataframe)

    def clear_table(self, table_name: str) -> None:
        """
        Clear all rows from the table specified by table_name, preserving the table schema.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' does not exist.")

        # Retrieve the current schema from the table.
        schema = self.tables[table_name].schema
        
        # Create an empty DataFrame with the same columns.
        empty_data = {col: [] for col in schema.keys()}
        new_df = pl.DataFrame(empty_data)
        
        # Cast the columns to their proper data types according to the schema.
        new_df = new_df.with_columns([pl.col(col).cast(dtype) for col, dtype in schema.items()])
        
        # Replace the existing table with the new, empty DataFrame.
        self.tables[table_name] = new_df

    def get_next_primary_key(self, table_name: str) -> int:
        """
        Get the next primary key (assumed to be in a column named "id") for the specified table.

        Parameters
        ----------
        table_name : str
            The name of the table.

        Returns
        -------
        int
            Next primary key (max id + 1) or 1 if table is empty or column does not exist.
        """
        df = self.get_table(table_name)
        if "id" in df.columns and df.height > 0:
            max_id = df["id"].max()
            return max_id + 1 if max_id is not None else 1
        return 1

class PolarsModel:
    _table_name: str = None  # Must be defined in subclasses

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate an object from a dictionary."""
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance

    def to_dict(self) -> dict:
        """Serialize the instance to a dictionary."""
        # You might want to filter out internal attributes (e.g., ones starting with '_')
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save(self, pdb: PolarsDB) -> None:
        """
        Save the model instance to the Polars DB.
        
        Automatically assigns a primary key (in column 'id') if not present.
        """
        if self._table_name is None:
            raise ValueError("Model must define a _table_name")
        if not hasattr(self, "id") or self.id is None:
            self.id = pdb.get_next_primary_key(self._table_name)
        pdb.add_row(self._table_name, self.to_dict())

class Dictionary2Job(PolarsModel):
    """
    Association model linking Dictionary and Job objects.
    
    This replaces the SQLAlchemy association table for dictionary-job mappings.
    
    Attributes
    ----------
    dictionary_id : int
        Identifier for the Dictionary.
    job_id : int
        Identifier for the Job.
    """
    _table_name = "dictionary_job"

    def __init__(self, dictionary_id: int, job_id: int, **kwargs):
        self.dictionary_id = dictionary_id
        self.job_id = job_id
        for key, value in kwargs.items():
            setattr(self, key, value)


class SpeakerOrdering(PolarsModel):
    """
    Association model for storing speaker ordering within a file.
    
    This replaces the SQLAlchemy association table 'speaker_ordering' that recorded
    the ordering of speakers in a file.
    
    Attributes
    ----------
    speaker_id : int
        Identifier for the Speaker.
    file_id : int
        Identifier for the File.
    index : int
        The order index for the speaker within the file.
    """
    _table_name = "speaker_ordering"

    def __init__(self, speaker_id: int, file_id: int, index: int, **kwargs):
        self.speaker_id = speaker_id
        self.file_id = file_id
        self.index = index
        for key, value in kwargs.items():
            setattr(self, key, value)

class Corpus(PolarsModel):
    """
    Polars version of the Corpus model for storing corpus information.

    This class replaces the SQLAlchemy-based Corpus model by leveraging the in-memory
    Polars DataFrame setup provided by PolarsDB. All modifications are kept in memory,
    and the 'save' method automatically assigns a primary key if not already set.

    Attributes
    ----------
    id : int | None
        The primary key, automatically assigned on save.
    name : str
        Corpus name.
    path : str
        Path to the corpus.
    imported : bool
        Flag indicating whether the corpus has been imported.
    text_normalized : bool
        Flag indicating whether text normalization is complete.
    cutoffs_found : bool
        Flag indicating whether cutoffs have been found.
    features_generated : bool
        Flag indicating whether feature extraction has been completed.
    vad_calculated : bool
        Flag indicating whether voice activity detection (VAD) has been computed.
    ivectors_calculated : bool
        Flag indicating whether ivector calculation is complete.
    plda_calculated : bool
        Flag indicating whether PLDA calculation is complete.
    xvectors_loaded : bool
        Flag indicating whether xvectors have been loaded.
    alignment_done : bool
        Flag indicating whether alignment processing is complete.
    transcription_done : bool
        Flag indicating whether transcription is complete.
    alignment_evaluation_done : bool
        Flag indicating whether alignment evaluation is complete.
    has_reference_alignments : bool
        Flag indicating whether reference alignments have been imported.
    has_sound_files : bool
        Flag indicating whether sound files exist.
    has_text_files : bool
        Flag indicating whether text files exist.
    num_jobs : int
        The number of jobs associated with this corpus.
    current_subset : int
        The current subset index.
    data_directory : str
        Directory under which the corpus data resides.
    """
    _table_name = "corpus"

    def __init__(
        self,
        name: str,
        path: typing.Union[str, Path],
        data_directory: typing.Union[str, Path],
        imported: bool = False,
        text_normalized: bool = False,
        cutoffs_found: bool = False,
        features_generated: bool = False,
        vad_calculated: bool = False,
        ivectors_calculated: bool = False,
        plda_calculated: bool = False,
        xvectors_loaded: bool = False,
        alignment_done: bool = False,
        transcription_done: bool = False,
        alignment_evaluation_done: bool = False,
        has_reference_alignments: bool = False,
        has_sound_files: bool = False,
        has_text_files: bool = False,
        num_jobs: int = 0,
        current_subset: int = 0,
        **kwargs,
    ):
        self.id = None  # Primary key will be assigned when saved via `save`
        self.name = name
        self.path = str(path) if isinstance(path, Path) else path
        self.data_directory = str(data_directory) if isinstance(data_directory, Path) else data_directory
        self.imported = imported
        self.text_normalized = text_normalized
        self.cutoffs_found = cutoffs_found
        self.features_generated = features_generated
        self.vad_calculated = vad_calculated
        self.ivectors_calculated = ivectors_calculated
        self.plda_calculated = plda_calculated
        self.xvectors_loaded = xvectors_loaded
        self.alignment_done = alignment_done
        self.transcription_done = transcription_done
        self.alignment_evaluation_done = alignment_evaluation_done
        self.has_reference_alignments = has_reference_alignments
        self.has_sound_files = has_sound_files
        self.has_text_files = has_text_files
        self.num_jobs = num_jobs
        self.current_subset = current_subset
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def split_directory(self) -> Path:
        """Return the split directory path based on the number of jobs."""
        return Path(self.data_directory).joinpath(f"split{self.num_jobs}")

    @property
    def current_subset_directory(self) -> Path:
        """Return the directory for the current subset."""
        if not self.current_subset:
            return self.split_directory
        return Path(self.data_directory).joinpath(f"subset_{self.current_subset}")

    @property
    def speaker_ivector_column(self) -> str:
        """
        Return the column name for the speaker ivector.
        Uses "xvector" if xvectors have been loaded; otherwise defaults to "ivector".
        """
        return "xvector" if self.xvectors_loaded else "ivector"

    @property
    def utterance_ivector_column(self) -> str:
        """
        Return the column name for the utterance ivector.
        Uses "xvector" if xvectors have been loaded; otherwise defaults to "ivector".
        """
        return "xvector" if self.xvectors_loaded else "ivector"
    
class Dialect(PolarsModel):
    """
    Polars version of the Dialect model for storing dialect information.

    Attributes
    ----------
    id : int | None
        The primary key (assigned automatically upon saving).
    name : str
        The name of the dialect.
    dictionaries : list
        List of associated dictionary records (in-memory).
    rules : list
        List of associated phonological rule records (in-memory).
    """
    _table_name = "dialect"

    def __init__(self, name: str, dictionaries: list = None, rules: list = None, **kwargs):
        self.id = None  # Primary key will be set on save
        self.name = name
        # Store relationships as in-memory lists:
        self.dictionaries = dictionaries if dictionaries is not None else []
        self.rules = rules if rules is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

class Dictionary(PolarsModel):
    """
    Polars version of the Dictionary model for storing pronunciation dictionary information.

    This model replaces the SQLAlchemy-based Dictionary by storing key attributes and
    associated relationships as in-memory Python objects. Relationships such as words,
    speakers, and jobs are maintained as lists, and helper properties (such as for building
    lexicons or symbol tables) are implemented to work without a SQLAlchemy session.

    Attributes
    ----------
    id : int | None
        Primary key (assigned automatically when saving via PolarsDB).
    name : str
        Dictionary name.
    path : str
        Path to the dictionary file/directory.
    rules_applied : bool
        Flag indicating if pronunciation rules have been applied.
    phone_set_type : str or None
        The phone set type (stored as a string).
    root_temp_directory : Path or None
        The root temporary directory (as a pathlib.Path) for building FSTs and other files.
    clitic_cleanup_regex : str or None
        Optional regex for cleaning up clitics.
    bracket_regex : str or None
        Regular expression for detecting bracketed words.
    laughter_regex : str or None
        Regular expression for detecting laughter words.
    position_dependent_phones : bool or None
        Flag for whether phones have position-dependent markers.
    default : bool
        Flag indicating if this dictionary is the default one.
    clitic_marker : str or None
        A single-character marker used for clitics.
    silence_word : str
        Symbol representing silence (default "<eps>").
    optional_silence_phone : str
        Symbol for the optional silence phone (default "sil").
    oov_word : str
        Symbol for out-of-vocabulary words (default "<unk>").
    oov_phone : str
        Symbol for the OOV phone (default "spn").
    bracketed_word : str or None
        Symbol for bracketed words (e.g., hesitations).
    cutoff_word : str or None
        Symbol used for cutoffs.
    laughter_word : str or None
        Symbol for laughter words.
    use_g2p : bool
        Flag indicating whether a G2P system is used (default False).
    max_disambiguation_symbol : int
        Maximum disambiguation index (default 0).
    silence_probability : float
        Probability of inserting non-initial optional silence (default 0.5).
    initial_silence_probability : float
        Probability of inserting initial silence (default 0.5).
    final_silence_correction : float or None
        Correction factor applied when final silence is present.
    final_non_silence_correction : float or None
        Correction factor applied when final non-silence occurs.
    dialect_id : int or None
        Identifier linking to the associated Dialect.
    dialect : object or None
        The associated Dialect object.
    words : list
        A list of associated Word objects.
    speakers : list
        A list of associated Speaker objects.
    jobs : list
        A list of associated Job objects.
    """
    _table_name = "dictionary"

    def __init__(
        self,
        name: str,
        path: typing.Union[str, Path],
        rules_applied: bool = False,
        phone_set_type: typing.Optional[str] = None,
        root_temp_directory: typing.Union[str, Path, None] = None,
        clitic_cleanup_regex: typing.Optional[str] = None,
        bracket_regex: typing.Optional[str] = None,
        laughter_regex: typing.Optional[str] = None,
        position_dependent_phones: typing.Optional[bool] = None,
        default: bool = False,
        clitic_marker: typing.Optional[str] = None,
        silence_word: str = "<eps>",
        optional_silence_phone: str = "sil",
        oov_word: str = "<unk>",
        oov_phone: str = "spn",
        bracketed_word: typing.Optional[str] = None,
        cutoff_word: typing.Optional[str] = None,
        laughter_word: typing.Optional[str] = None,
        use_g2p: bool = False,
        max_disambiguation_symbol: int = 0,
        silence_probability: float = 0.5,
        initial_silence_probability: float = 0.5,
        final_silence_correction: typing.Optional[float] = None,
        final_non_silence_correction: typing.Optional[float] = None,
        dialect_id: typing.Optional[int] = None,
        dialect=None,
        words: list = None,
        speakers: list = None,
        jobs: list = None,
        **kwargs,
    ):
        self.id = None  # Primary key will be assigned when saved via the PolarsDB.
        self.name = name
        self.path = str(path) if isinstance(path, Path) else path
        self.rules_applied = rules_applied
        self.phone_set_type = phone_set_type
        self.root_temp_directory = Path(root_temp_directory) if root_temp_directory else None
        self.clitic_cleanup_regex = clitic_cleanup_regex
        self.bracket_regex = bracket_regex
        self.laughter_regex = laughter_regex
        self.position_dependent_phones = position_dependent_phones
        self.default = default
        self.clitic_marker = clitic_marker
        self.silence_word = silence_word
        self.optional_silence_phone = optional_silence_phone
        self.oov_word = oov_word
        self.oov_phone = oov_phone
        self.bracketed_word = bracketed_word
        self.cutoff_word = cutoff_word
        self.laughter_word = laughter_word
        self.use_g2p = use_g2p
        self.max_disambiguation_symbol = max_disambiguation_symbol
        self.silence_probability = silence_probability
        self.initial_silence_probability = initial_silence_probability
        self.final_silence_correction = final_silence_correction
        self.final_non_silence_correction = final_non_silence_correction
        self.dialect_id = dialect_id
        self.dialect = dialect
        self.words = words if words is not None else []
        self.speakers = speakers if speakers is not None else []
        self.jobs = jobs if jobs is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def word_mapping(self) -> dict:
        """
        Compute a mapping from word to mapping_id for included words.

        For non-default dictionaries, filters words whose dictionary_id matches self.id.
        For the default dictionary, groups by unique (word, mapping_id) pairs.
        """
        if not hasattr(self, "_word_mapping"):
            if self.words:
                if self.name != "default":
                    filtered = [
                        w for w in self.words
                        if getattr(w, "included", False)
                        and getattr(w, "dictionary_id", None) == self.id
                    ]
                else:
                    seen = {}
                    for w in self.words:
                        if getattr(w, "included", False):
                            key = (w.word, w.mapping_id)
                            seen[key] = w
                    filtered = list(seen.values())
                filtered.sort(key=lambda x: x.mapping_id)
                self._word_mapping = {w.word: w.mapping_id for w in filtered}
            else:
                self._word_mapping = {}
        return self._word_mapping

    @property
    def word_table(self):
        """
        Build and return a SymbolTable for words.

        If the file at self.words_symbol_path exists, it is loaded.
        Otherwise, a new symbol table is built from self.words and written to file.
        """
        if not hasattr(self, "_word_table"):
            from pywrapfst import SymbolTable  # assuming pywrapfst is available
            path = self.words_symbol_path
            if path.exists():
                self._word_table = SymbolTable.read_text(str(path))
            else:
                self.temp_directory.mkdir(parents=True, exist_ok=True)
                self._word_table = SymbolTable()
                if self.words:
                    if self.name != "default":
                        filtered = [
                            w for w in self.words
                            if getattr(w, "included", False)
                            and getattr(w, "dictionary_id", None) == self.id
                        ]
                    else:
                        seen = {}
                        for w in self.words:
                            if getattr(w, "included", False):
                                key = (w.word, w.mapping_id)
                                seen[key] = w
                        filtered = list(seen.values())
                    filtered.sort(key=lambda x: x.mapping_id)
                    for w in filtered:
                        self._word_table.add_symbol(w.word, w.mapping_id)
                self._word_table.write_text(str(path))
        return self._word_table

    @property
    def phone_table(self):
        """
        Build and return a SymbolTable for phones.

        If the file at self.phone_symbol_table_path exists, it is loaded
        (ensuring special symbols are present). Otherwise, the table is built from self.phones.
        """
        if not hasattr(self, "_phone_table"):
            from pywrapfst import SymbolTable  # assuming pywrapfst is available
            path = self.phone_symbol_table_path
            if path.exists():
                self._phone_table = SymbolTable.read_text(str(path))
                for k in ["#0", "#1", "#2"]:
                    if not self._phone_table.member(k):
                        self._phone_table.add_symbol(k)
            else:
                self.phones_directory.mkdir(parents=True, exist_ok=True)
                self._phone_table = SymbolTable()
                # Assume self.phones is a list; if not available, use an empty list.
                phones = self.phones if hasattr(self, "phones") and self.phones is not None else []
                phones.sort(key=lambda p: p.mapping_id)
                for p in phones:
                    self._phone_table.add_symbol(p.kaldi_label, p.mapping_id)
                self._phone_table.write_text(str(path))
        return self._phone_table

    @property
    def word_pronunciations(self):
        """
        Construct a mapping from words to a set of pronunciations.
        """
        if not hasattr(self, "_word_pronunciations"):
            if self.words:
                pairs = []
                for w in self.words:
                    if getattr(w, "included", False):
                        # Assume each word object has a list attribute 'pronunciations'
                        for p in getattr(w, "pronunciations", []):
                            if p.pronunciation != self.oov_phone:
                                pairs.append((w.word, p.pronunciation))
                if self.name != "default":
                    pairs = [
                        (w, p) for w, p in pairs
                        if getattr(w, "dictionary_id", None) == self.id
                    ]
                else:
                    pairs = list({(w, p) for w, p in pairs})
                mapping = {}
                for word, pron in pairs:
                    mapping.setdefault(word, set()).add(pron)
                self._word_pronunciations = mapping
            else:
                self._word_pronunciations = {}
        return self._word_pronunciations

    @property
    def lexicon_compiler(self):
        """
        Construct and return a LexiconCompiler object configured with dictionary parameters.
        """
        lexicon_compiler = LexiconCompiler(
            silence_probability=self.silence_probability,
            initial_silence_probability=self.initial_silence_probability,
            final_silence_correction=self.final_silence_correction,
            final_non_silence_correction=self.final_non_silence_correction,
            silence_word=self.silence_word,
            oov_word=self.oov_word,
            silence_phone=self.optional_silence_phone,
            oov_phone=self.oov_phone,
            position_dependent_phones=self.position_dependent_phones,
        )
        if self.lexicon_disambig_fst_path.exists():
            lexicon_compiler.load_l_from_file(str(self.lexicon_disambig_fst_path))
            lexicon_compiler.disambiguation = True
        elif self.lexicon_fst_path.exists():
            lexicon_compiler.load_l_from_file(str(self.lexicon_fst_path))
        if self.align_lexicon_disambig_path.exists():
            lexicon_compiler.load_l_align_from_file(str(self.align_lexicon_disambig_path))
        elif self.align_lexicon_path.exists():
            lexicon_compiler.load_l_align_from_file(str(self.align_lexicon_path))
        lexicon_compiler.word_table = self.word_table
        lexicon_compiler.phone_table = self.phone_table
        return lexicon_compiler

    @property
    def special_set(self) -> typing.Set[str]:
        """
        Returns a set containing special symbols.
        """
        return {"<s>", "</s>", self.silence_word, self.oov_word, self.bracketed_word, self.laughter_word}

    @property
    def clitic_set(self) -> typing.Set[str]:
        """
        Returns a set of clitic words based on associated words.
        """
        return {x.word for x in self.words if hasattr(x, "word_type") and x.word_type == WordType.clitic}

    @property
    def word_boundary_int_path(self) -> Path:
        """
        Path to the word boundary integer IDs.
        """
        return self.phones_directory.joinpath("word_boundary.int")

    @property
    def disambiguation_symbols_int_path(self) -> Path:
        """
        Path to the disambiguation symbols integer IDs.
        """
        return self.phones_directory.joinpath("disambiguation_symbols.int")

    @property
    def phones_directory(self) -> Path:
        """
        Directory containing phone-related files.
        """
        return self.root_temp_directory.joinpath("phones") if self.root_temp_directory else Path(".")

    @property
    def phone_symbol_table_path(self) -> Path:
        """
        Path to the phone symbol table file.
        """
        return self.phones_directory.joinpath("phones.txt")

    @property
    def grapheme_symbol_table_path(self) -> Path:
        """
        Path to the grapheme symbol table file.
        """
        return self.phones_directory.joinpath("graphemes.txt")

    @property
    def phone_disambig_path(self) -> Path:
        """
        Path to the phone disambiguation file.
        """
        return self.phones_directory.joinpath("phone_disambig.txt")

    @property
    def temp_directory(self) -> Path:
        """
        Temporary directory for dictionary-specific files.
        """
        return self.root_temp_directory.joinpath(f"{self.id}_{self.name}") if self.root_temp_directory else Path(".")

    @property
    def lexicon_disambig_fst_path(self) -> Path:
        """
        Path of the disambiguated lexicon FST.
        """
        return self.temp_directory.joinpath("L.disambig_fst")

    @property
    def align_lexicon_path(self) -> Path:
        """
        Path of the lexicon FST used for aligning lattices.
        """
        return self.temp_directory.joinpath("align_lexicon.fst")

    @property
    def align_lexicon_disambig_path(self) -> Path:
        """
        Path of the disambiguated lexicon FST used for aligning lattices.
        """
        return self.temp_directory.joinpath("align_lexicon.disambig_fst")

    @property
    def align_lexicon_int_path(self) -> Path:
        """
        Path of the integer version of the aligned lexicon.
        """
        return self.temp_directory.joinpath("align_lexicon.int")

    @property
    def lexicon_fst_path(self) -> Path:
        """
        Path of the lexicon FST.
        """
        return self.temp_directory.joinpath("L.fst")

    @property
    def words_symbol_path(self) -> Path:
        """
        Path to the word-symbol mapping file for the dictionary.
        """
        return self.temp_directory.joinpath("words.txt")

    @property
    def data_source_identifier(self) -> str:
        """
        Unique identifier for the dictionary based on its id and name.
        """
        return f"{self.id}_{self.name}"

    @property
    def identifier(self) -> str:
        """
        Alias for data_source_identifier.
        """
        return self.data_source_identifier

    @property
    def silence_probability_info(self) -> typing.Dict[str, float]:
        """
        Returns a dictionary of silence probability information.
        """
        return {
            "silence_probability": self.silence_probability,
            "initial_silence_probability": self.initial_silence_probability,
            "final_silence_correction": self.final_silence_correction,
            "final_non_silence_correction": self.final_non_silence_correction,
        }

class Phone(PolarsModel):
    """
    Polars version of the Phone model for storing phones and their integer IDs.

    Attributes
    ----------
    id : int | None
        Auto-assigned primary key (set when saved).
    mapping_id : int
        Integer ID of the phone for Kaldi processing.
    phone : str
        Phone label.
    kaldi_label : str
        Unique Kaldi label for the phone.
    position : str | None
        Optional position information.
    phone_type : any
        Type of phone.
    mean_duration : float | None
        Mean duration of the phone.
    sd_duration : float | None
        Standard deviation of the phone's duration.
    phone_intervals : list
        List of associated PhoneInterval objects.
    """

    _table_name = "phone"

    def __init__(
        self,
        mapping_id: int,
        phone: str,
        kaldi_label: str,
        position: typing.Optional[str] = None,
        phone_type: typing.Any = None,
        mean_duration: typing.Optional[float] = None,
        sd_duration: typing.Optional[float] = None,
        phone_intervals: list = None,
        **kwargs,
    ):
        self.id = None  # Primary key is assigned upon saving.
        self.mapping_id = mapping_id
        self.phone = phone
        self.kaldi_label = kaldi_label
        self.position = position
        self.phone_type = phone_type
        self.mean_duration = mean_duration
        self.sd_duration = sd_duration
        self.phone_intervals = phone_intervals if phone_intervals is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)


class Grapheme(PolarsModel):
    """
    Polars version of the Grapheme model for storing graphemes and their integer IDs.

    Attributes
    ----------
    id : int | None
        Auto-assigned primary key (set when saved).
    mapping_id : int
        Integer ID of the grapheme.
    grapheme : str
        Grapheme label.
    """

    _table_name = "grapheme"

    def __init__(self, mapping_id: int, grapheme: str, **kwargs):
        self.id = None  # Primary key is assigned upon saving.
        self.mapping_id = mapping_id
        self.grapheme = grapheme
        for key, value in kwargs.items():
            setattr(self, key, value)


class Word(PolarsModel):
    """
    Polars version of the Word model for storing words, their integer IDs,
    and pronunciation associations.

    Attributes
    ----------
    id : int | None
        Auto-assigned primary key (set when saved).
    mapping_id : int
        Integer ID of the word for Kaldi processing.
    word : str
        The word label.
    count : int
        Frequency count of the word.
    word_type : any
        Type of the word (can be an enum value or a string).
    included : bool
        Flag indicating if the word is included.
    initial_cost : float | None
        Optional initial cost.
    final_cost : float | None
        Optional final cost.
    dictionary_id : int | None
        Identifier linking to the corresponding Dictionary.
    dictionary : object | None
        The associated Dictionary object.
    pronunciations : list
        List of associated Pronunciation objects.
    word_intervals : list
        List of associated WordInterval objects.
    job : object | None
        Associated Word2Job object (if any).
    """

    _table_name = "word"

    def __init__(
        self,
        mapping_id: int,
        word: str,
        word_type: typing.Any,
        count: int = 0,
        included: bool = True,
        initial_cost: typing.Optional[float] = None,
        final_cost: typing.Optional[float] = None,
        dictionary_id: typing.Optional[int] = None,
        dictionary=None,
        pronunciations: list = None,
        word_intervals: list = None,
        job=None,
        **kwargs,
    ):
        self.id = None  # Primary key is assigned upon saving.
        self.mapping_id = mapping_id
        self.word = word
        self.count = count
        self.word_type = word_type
        self.included = included
        self.initial_cost = initial_cost
        self.final_cost = final_cost
        self.dictionary_id = dictionary_id
        self.dictionary = dictionary
        self.pronunciations = pronunciations if pronunciations is not None else []
        self.word_intervals = word_intervals if word_intervals is not None else []
        self.job = job
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        d = super().to_dict()
        if isinstance(self.word_type, WordType):
            d["word_type"] = self.word_type.value
        return d

class Pronunciation(PolarsModel):
    _table_name = "pronunciation"

    def __init__(
        self,
        pronunciation: str,
        probability: float = None,
        disambiguation: int = None,
        silence_after_probability: float = None,
        silence_before_correction: float = None,
        non_silence_before_correction: float = None,
        generated_by_rule: bool = False,
        count: int = 0,
        silence_following_count: int = None,
        non_silence_following_count: int = None,
        word_id: int = None,
        word=None,
        rules: list = None,
        word_intervals: list = None,
        **kwargs,
    ):
        """
        Initialize a Pronunciation instance.
        
        Parameters
        ----------
        pronunciation : str
            Space-delimited pronunciation.
        probability : float, optional
            Probability of the pronunciation.
        disambiguation : int, optional
            Disambiguation index.
        silence_after_probability : float, optional
            Probability of silence following the pronunciation.
        silence_before_correction : float, optional
            Correction factor for silence before the pronunciation.
        non_silence_before_correction : float, optional
            Correction factor for non-silence before the pronunciation.
        generated_by_rule : bool, optional
            Flag indicating if the pronunciation was generated by a rule.
        count : int, optional
            Count associated with the pronunciation.
        silence_following_count : int, optional
            Count of silence following the pronunciation.
        non_silence_following_count : int, optional
            Count of non-silence following the pronunciation.
        word_id : int, optional
            Identifier for the associated word.
        word : object, optional
            Associated word instance or its data.
        rules : list, optional
            List of rule applications that involve this pronunciation.
        word_intervals : list, optional
            List of word intervals associated with this pronunciation.
        kwargs :
            Any additional keyword arguments.
        """
        self.id = None
        self.pronunciation = pronunciation
        self.probability = probability
        self.disambiguation = disambiguation
        self.silence_after_probability = silence_after_probability
        self.silence_before_correction = silence_before_correction
        self.non_silence_before_correction = non_silence_before_correction
        self.generated_by_rule = generated_by_rule
        self.count = count
        self.silence_following_count = silence_following_count
        self.non_silence_following_count = non_silence_following_count
        self.word_id = word_id
        self.word = word
        self.rules = rules if rules is not None else []
        self.word_intervals = word_intervals if word_intervals is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

class PhonologicalRule(PolarsModel):
    _table_name = "phonological_rule"

    def __init__(
        self,
        segment: str,
        preceding_context: str,
        following_context: str,
        replacement: str,
        probability: float = None,
        silence_after_probability: float = None,
        silence_before_correction: float = None,
        non_silence_before_correction: float = None,
        dialect_id: int = None,
        dialect=None,  # You can store an associated dialect instance or simply its id
        **kwargs,
    ):
        self.id = None
        self.segment = segment
        self.preceding_context = preceding_context
        self.following_context = following_context
        self.replacement = replacement
        self.probability = probability
        self.silence_after_probability = silence_after_probability
        self.silence_before_correction = silence_before_correction
        self.non_silence_before_correction = non_silence_before_correction
        self.dialect_id = dialect_id
        self.dialect = dialect
        # Additional keyword arguments can be stored as extra attributes:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __hash__(self):
        return hash((self.segment, self.preceding_context, self.following_context, self.replacement))

    def to_json(self) -> dict:
        """
        Serializes the rule for export.
        """
        return {
            "segment": self.segment,
            "dialect": self.dialect,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "replacement": self.replacement,
            "probability": self.probability,
            "silence_after_probability": self.silence_after_probability,
            "silence_before_correction": self.silence_before_correction,
            "non_silence_before_correction": self.non_silence_before_correction,
        }

    @property
    def match_regex(self):
        """
        Returns a regular expression compiled from the rule contents.
        """
        components = []
        initial = False
        final = False
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            initial = True
            preceding = preceding.replace("^", "").strip()
        if following.endswith("$"):
            final = True
            following = following.replace("$", "").strip()
        if preceding:
            components.append(rf"(?P<preceding>{preceding})")
        if self.segment:
            components.append(rf"(?P<segment>{self.segment})")
        if following:
            components.append(rf"(?P<following>{following})")
        pattern = " ".join(components)
        if initial:
            pattern = "^" + pattern
        else:
            pattern = r"(?:^|(?<=\s))" + pattern
        if final:
            pattern += "$"
        else:
            pattern += r"(?:$|(?=\s))"
        return re.compile(pattern, flags=re.UNICODE)

    def __str__(self):
        """
        Returns a string representation of the rule.
        """
        from_components = []
        to_components = []
        initial = False
        final = False
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            initial = True
            preceding = preceding.replace("^", "").strip()
        if following.endswith("$"):
            final = True
            following = following.replace("$", "").strip()
        if preceding:
            from_components.append(preceding)
            to_components.append(preceding)
        if self.segment:
            from_components.append(self.segment)
        if self.replacement:
            to_components.append(self.replacement)
        if following:
            from_components.append(following)
            to_components.append(following)

        from_string = " ".join(from_components)
        to_string = " ".join(to_components)
        if initial:
            from_string = "^" + from_string
        if final:
            from_string += "$"
        return f"<PhonologicalRule {self.id} for Dialect {self.dialect_id}: {from_string} -> {to_string}>"

    def apply_rule(self, pronunciation: str) -> str:
        """
        Apply the rule on a pronunciation by replacing any matching segments with the replacement.
        """
        preceding = self.preceding_context
        following = self.following_context
        if preceding.startswith("^"):
            preceding = preceding.replace("^", "").strip()
        if following.startswith("$"):
            following = following.replace("$", "").strip()
        components = []
        if preceding:
            components.append(r"\g<preceding>")
        if self.replacement:
            components.append(self.replacement)
        if following:
            components.append(r"\g<following>")
        return self.match_regex.sub(" ".join(components), pronunciation).strip()

class RuleApplication(PolarsModel):
    _table_name = "rule_applications"

    def __init__(
        self,
        pronunciation_id: int,
        rule_id: int,
        pronunciation=None,
        rule=None,
        **kwargs,
    ):
        """
        Initialize a RuleApplication instance.
        
        Parameters
        ----------
        pronunciation_id : int
            Identifier for the associated Pronunciation.
        rule_id : int
            Identifier for the associated PhonologicalRule.
        pronunciation : object, optional
            The associated Pronunciation instance or its data.
        rule : object, optional
            The associated PhonologicalRule instance or its data.
        kwargs :
            Any additional keyword arguments.
        """
        self.pronunciation_id = pronunciation_id
        self.rule_id = rule_id
        self.pronunciation = pronunciation
        self.rule = rule
        for key, value in kwargs.items():
            setattr(self, key, value)

class Speaker(PolarsModel):
    _table_name = "speaker"

    def __init__(
        self,
        name: str,
        cmvn: str = None,
        fmllr: str = None,
        min_f0: float = None,
        max_f0: float = None,
        ivector = None,  # Store ivector as-is (e.g., a list or other type)
        plda_vector = None,
        xvector = None,
        num_utterances: int = None,
        modified: bool = False,
        dictionary_id: int = None,
        dictionary=None,
        utterances: list = None,
        files: list = None,
        **kwargs,
    ):
        """
        Initialize a Speaker instance.

        Parameters
        ----------
        name : str
            Name of the speaker.
        cmvn : str, optional
            File index for the speaker's CMVN stats.
        fmllr : str, optional
            FMLLR information for the speaker.
        min_f0 : float, optional
            Minimum F0 for the speaker.
        max_f0 : float, optional
            Maximum F0 for the speaker.
        ivector : optional
            Ivector data.
        plda_vector : optional
            PLDA vector data.
        xvector : optional
            Xvector data.
        num_utterances : int, optional
            Number of utterances associated with the speaker.
        modified : bool, optional
            Flag indicating if the speaker has been modified.
        dictionary_id : int, optional
            Identifier for the associated dictionary.
        dictionary : object, optional
            The associated dictionary instance or its data.
        utterances : list, optional
            A list of associated utterance objects.
        files : list, optional
            A list of file records in which the speaker appears.
        kwargs :
            Any additional keyword arguments.
        """
        self.id = None
        self.name = name
        self.cmvn = cmvn
        self.fmllr = fmllr
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.ivector = ivector
        self.plda_vector = plda_vector
        self.xvector = xvector
        self.num_utterances = num_utterances
        self.modified = modified
        self.dictionary_id = dictionary_id
        self.dictionary = dictionary
        # Initialize relationships as empty lists if not provided
        self.utterances = utterances if utterances is not None else []
        self.files = files if files is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

class Grapheme(PolarsModel):
    _table_name = "grapheme"

    def __init__(
        self,
        grapheme: str,
        mapping_id: int,
        **kwargs,
    ):
        self.id = None
        self.grapheme = grapheme
        self.mapping_id = mapping_id
        for key, value in kwargs.items():
            setattr(self, key, value)

class Phone(PolarsModel):
    _table_name = "phone"

    def __init__(
        self,
        phone: str,
        mapping_id: int,
        kaldi_label: str,
        phone_type: int,
        position: str = None,
        mean_duration: float = None,
        sd_duration: float = None,
        **kwargs,
    ):
        self.id = None
        self.phone = phone
        self.mapping_id = mapping_id
        self.kaldi_label = kaldi_label
        self.phone_type = phone_type
        self.position = position
        self.mean_duration = mean_duration
        self.sd_duration = sd_duration
        for key, value in kwargs.items():
            setattr(self, key, value)

class File(PolarsModel):
    _table_name = "file"

    def __init__(
        self,
        name: str,
        relative_path: typing.Union[Path, str],
        modified: bool = False,
        speakers: typing.List = None,
        text_file=None,
        sound_file=None,
        utterances: typing.List = None,
        **kwargs,
    ):
        """
        Initialize a File instance.

        Parameters
        ----------
        name : str
            Base name of the file.
        relative_path : Path or str
            File path relative to the corpus root.
        modified : bool, optional
            Flag indicating if the file has been modified.
        speakers : list, optional
            List of speaker objects in the file.
        text_file : TextFile, optional
            Associated transcription file.
        sound_file : SoundFile, optional
            Associated audio file.
        utterances : list, optional
            List of utterance objects in the file.
        kwargs :
            Additional keyword arguments.
        """
        self.id = None
        self.name = name
        self.relative_path = relative_path if isinstance(relative_path, Path) else Path(relative_path)
        self.modified = modified
        self.speakers = speakers if speakers is not None else []
        self.text_file = text_file
        self.sound_file = sound_file
        self.utterances = utterances if utterances is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_speakers(self) -> int:
        """Return the number of speakers in the file."""
        return len(self.speakers)

    @property
    def num_utterances(self) -> int:
        """Return the number of utterances in the file."""
        return len(self.utterances)

    @property
    def duration(self) -> float:
        """Return the duration of the associated sound file."""
        return self.sound_file.duration if self.sound_file else 0.0

    @property
    def num_channels(self) -> int:
        """Return the number of channels from the associated sound file."""
        return self.sound_file.num_channels if self.sound_file else 0

    @property
    def sample_rate(self) -> int:
        """Return the sample rate from the associated sound file."""
        return self.sound_file.sample_rate if self.sound_file else 0

    def save(
        self,
        output_directory: typing.Union[str, Path],
        output_format: typing.Optional[str] = None,
        save_transcription: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Output the file to a TextGrid or LAB file. If output_format is not specified,
        the original file type is used or guessed based on the utterance boundary values.

        Parameters
        ----------
        output_directory : str or Path
            Directory to write the output file.
        output_format : str, optional
            Format in which to save (e.g., "lab" or "TextGrid").
        save_transcription : bool, optional
            If True, save the hypothesized transcription rather than the default text.
        overwrite : bool, optional
            If True, overwrite existing output.
        """
        from montreal_forced_aligner.textgrid import construct_output_path

        output_directory = (
            output_directory if isinstance(output_directory, Path) else Path(output_directory)
        )
        utterance_count = len(self.utterances)
        if output_format is None:
            if (utterance_count == 1 and self.utterances[0].begin == 0 and 
                self.utterances[0].end == self.duration):
                output_format = TextFileType.LAB.value
            else:
                output_format = TextFileType.TEXTGRID.value

        output_path = construct_output_path(
            self.name, self.relative_path, output_directory, output_format=output_format
        )
        if overwrite:
            if self.text_file is None:
                self.text_file = TextFile(file_id=self.id, text_file_path=output_path, file_type=output_format)
            if output_path != self.text_file.text_file_path and os.path.exists(self.text_file.text_file_path):
                os.remove(self.text_file.text_file_path)
            self.text_file.file_type = output_format
            self.text_file.text_file_path = output_path

        if output_format == TextFileType.LAB.value:
            if utterance_count == 0:
                if self.text_file and os.path.exists(self.text_file.text_file_path) and not save_transcription:
                    os.remove(self.text_file.text_file_path)
                return
            with mfa_open(output_path, "w") as f:
                for u in self.utterances:
                    if save_transcription:
                        f.write(u.transcription_text if u.transcription_text else "")
                    elif u.text:
                        f.write(u.text)
            return
        elif output_format == TextFileType.TEXTGRID.value:
            max_time = self.sound_file.duration if self.sound_file else 0.0
            tiers = {}
            for speaker in self.speakers:
                tiers[speaker.name] = textgrid.IntervalTier(speaker.name, [], minT=0, maxT=max_time)

            tg = textgrid.Textgrid()
            tg.maxTimestamp = max_time
            for utterance in self.utterances:
                spk_name = utterance.speaker.name if hasattr(utterance, 'speaker') else "Unknown"
                if spk_name not in tiers:
                    tiers[spk_name] = textgrid.IntervalTier(spk_name, [], minT=0, maxT=max_time)
                if save_transcription:
                    tiers[spk_name].insertEntry(
                        Interval(
                            start=utterance.begin,
                            end=utterance.end,
                            label=utterance.transcription_text or ""
                        )
                    )
                else:
                    if utterance.end < utterance.begin:
                        utterance.begin, utterance.end = utterance.end, utterance.begin
                    if tiers[spk_name].entries:
                        if tiers[spk_name].entries[-1].end > utterance.begin:
                            utterance.begin = tiers[spk_name].entries[-1].end
                    if utterance.end > self.duration:
                        utterance.end = self.duration
                    tiers[spk_name].insertEntry(
                        Interval(
                            start=utterance.begin,
                            end=utterance.end,
                            label=utterance.text.strip()
                        )
                    )
            for t in tiers.values():
                tg.addTier(t)
            tg.save(output_path, includeBlankSpaces=True, format=output_format)

    def construct_transcription_tiers(
        self, original_text: bool = False
    ) -> typing.Dict[str, typing.Dict[str, typing.List[CtmInterval]]]:
        """
        Construct transcription tiers based on utterances.

        Parameters
        ----------
        original_text : bool, optional
            If True, use the original text; otherwise, use the transcription text.

        Returns
        -------
        dict[str, dict[str, list[CtmInterval]]]
            A tier dictionary where keys are speaker names.
        """
        data = {}
        for u in self.utterances:
            speaker_name = u.speaker_name if hasattr(u, 'speaker_name') else "Unknown"
            data.setdefault(speaker_name, {})
            if original_text:
                label = u.text
                key = "text"
            else:
                label = u.transcription_text
                key = "transcription"
            if not label:
                label = ""
            data[speaker_name].setdefault(key, []).append(CtmInterval(u.begin, u.end, label))
        return data


class SoundFile(PolarsModel):
    _table_name = "sound_file"

    def __init__(
        self,
        file_id: int,
        sound_file_path: typing.Union[Path, str],
        format: str,
        sample_rate: int,
        duration: float,
        num_channels: int,
        sox_string: str = None,
        file=None,
        **kwargs,
    ):
        """
        Initialize a SoundFile instance.

        Parameters
        ----------
        file_id : int
            Identifier linking to the associated File.
        sound_file_path : Path or str
            Path to the audio file.
        format : str
            Audio file format (e.g., "wav", "flac").
        sample_rate : int
            Sample rate of the audio file.
        duration : float
            Duration of the audio file.
        num_channels : int
            Number of channels.
        sox_string : str, optional
            Sox processing string.
        file : File, optional
            Associated File object.
        kwargs :
            Additional keyword arguments.
        """
        self.file_id = file_id
        self.file = file
        self.sound_file_path = sound_file_path if isinstance(sound_file_path, Path) else Path(sound_file_path)
        self.format = format
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_channels = num_channels
        self.sox_string = sox_string
        for key, value in kwargs.items():
            setattr(self, key, value)

    def normalized_waveform(
        self, begin: float = 0, end: typing.Optional[float] = None
    ) -> typing.Tuple[np.array, np.array]:
        """
        Load and return a normalized waveform from the audio file.

        Parameters
        ----------
        begin : float, optional
            Start time in seconds (default is 0).
        end : float, optional
            End time in seconds (default is the file duration).

        Returns
        -------
        tuple of np.array
            A tuple (time_points, normalized_samples).
        """
        if end is None or end > self.duration:
            end = self.duration

        y, _ = librosa.load(
            str(self.sound_file_path), sr=None, mono=False, offset=begin, duration=end - begin
        )
        if y.ndim > 1 and y.shape[0] == 2:
            y = y / np.max(np.abs(y))
            num_steps = y.shape[1]
        else:
            y = y / np.max(np.abs(y), axis=0)
            num_steps = y.shape[0]
        y[np.isnan(y)] = 0
        x = np.linspace(start=begin, stop=end, num=num_steps)
        return x, y

    def load_audio(
        self, begin: float = 0, end: typing.Optional[float] = None
    ) -> typing.Tuple[np.array, np.array]:
        """
        Load and return the audio waveform for processing.

        Parameters
        ----------
        begin : float, optional
            Start time in seconds (default is 0).
        end : float, optional
            End time in seconds (default is the file duration).

        Returns
        -------
        tuple of np.array
            A tuple (audio_samples, time_points).
        """
        if end is None or end > self.duration:
            end = self.duration

        y, _ = librosa.load(
            str(self.sound_file_path), sr=16000, mono=False, offset=begin, duration=end - begin
        )
        return y

class TextFile(PolarsModel):
    _table_name = "text_file"

    def __init__(
        self,
        file_id: int,
        text_file_path: typing.Union[Path, str],
        file_type: str,
        file=None,
        **kwargs,
    ):
        """
        Initialize a TextFile instance.

        Parameters
        ----------
        file_id : int
            Identifier linking to the File.
        text_file_path : Path or str
            Path to the transcription file.
        file_type : str
            Type of transcription file (e.g., "lab", "TextGrid").
        file : File, optional
            Associated File object.
        kwargs :
            Additional keyword arguments.
        """
        self.file_id = file_id
        self.file = file
        self.text_file_path = text_file_path if isinstance(text_file_path, Path) else Path(text_file_path)
        self.file_type = file_type
        for key, value in kwargs.items():
            setattr(self, key, value)

class Utterance(PolarsModel):
    """
    Polars version of the Utterance model for storing utterance information in-memory.

    Attributes
    ----------
    id : int | None
        Auto-assigned primary key (set when saved).
    begin : float
        Beginning timestamp of the utterance.
    end : float
        Ending timestamp of the utterance.
    channel : int
        Channel of the utterance in the audio file.
    num_frames : int | None
        Number of feature frames extracted.
    text : str | None
        Input text for the utterance.
    oovs : str | None
        Space-delimited list of out-of-vocabulary items.
    normalized_text : str | None
        Normalized text for the utterance.
    normalized_character_text : str | None
        Normalized character text.
    transcription_text : str | None
        Transcription text.
    features : str | None
        File index for generated features.
    ivector_ark : str | None
        Path to the ivector ark file.
    vad_ark : str | None
        Path to the vad ark file.
    in_subset : bool
        Flag indicating if the utterance is in the current training subset.
    ignored : bool
        Flag indicating whether the utterance is ignored (e.g. due to missing features).
    alignment_log_likelihood : float | None
        Log likelihood for the complete alignment.
    speech_log_likelihood : float | None
        Log likelihood computed using only speech phones.
    duration_deviation : float | None
        Average of the absolute z-scores for speech phone durations.
    phone_error_rate : float | None
        Phone error rate for alignment evaluation.
    alignment_score : float | None
        Alignment score computed during evaluation.
    word_error_rate : float | None
        Word error rate for transcription evaluation.
    character_error_rate : float | None
        Character error rate for transcription evaluation.
    ivector : any | None
        Ivector representation.
    plda_vector : any | None
        PLDA vector.
    xvector : any | None
        Xvector representation.
    file_id : int | None
        Identifier of the associated File.
    speaker_id : int | None
        Identifier of the associated Speaker.
    job_id : int | None
        Identifier of the Job that processes the utterance.
    file : object | None
        Associated File object.
    speaker : object | None
        Associated Speaker object.
    job : object | None
        Associated Job object.
    phone_intervals : list
        List of associated PhoneInterval objects.
    word_intervals : list
        List of associated WordInterval objects.
    """
    _table_name = "utterance"

    def __init__(
        self,
        begin: float,
        end: float,
        channel: int,
        num_frames: typing.Optional[int] = None,
        text: typing.Optional[str] = None,
        oovs: typing.Optional[str] = None,
        normalized_text: typing.Optional[str] = None,
        normalized_character_text: typing.Optional[str] = None,
        transcription_text: typing.Optional[str] = None,
        features: typing.Optional[str] = None,
        ivector_ark: typing.Optional[str] = None,
        vad_ark: typing.Optional[str] = None,
        in_subset: bool = False,
        ignored: bool = False,
        alignment_log_likelihood: typing.Optional[float] = None,
        speech_log_likelihood: typing.Optional[float] = None,
        duration_deviation: typing.Optional[float] = None,
        phone_error_rate: typing.Optional[float] = None,
        alignment_score: typing.Optional[float] = None,
        word_error_rate: typing.Optional[float] = None,
        character_error_rate: typing.Optional[float] = None,
        ivector=None,
        plda_vector=None,
        xvector=None,
        file_id: typing.Optional[int] = None,
        speaker_id: typing.Optional[int] = None,
        job_id: typing.Optional[int] = None,
        file=None,
        speaker=None,
        job=None,
        phone_intervals: list = None,
        word_intervals: list = None,
        **kwargs,
    ):
        self.id = None  # Primary key is assigned upon saving.
        self.begin = begin
        self.end = end
        self.channel = channel
        self.num_frames = num_frames
        self.text = text
        self.oovs = oovs
        self.normalized_text = normalized_text
        self.normalized_character_text = normalized_character_text
        self.transcription_text = transcription_text
        self.features = features
        self.ivector_ark = ivector_ark
        self.vad_ark = vad_ark
        self.in_subset = in_subset
        self.ignored = ignored
        self.alignment_log_likelihood = alignment_log_likelihood
        self.speech_log_likelihood = speech_log_likelihood
        self.duration_deviation = duration_deviation
        self.phone_error_rate = phone_error_rate
        self.alignment_score = alignment_score
        self.word_error_rate = word_error_rate
        self.character_error_rate = character_error_rate
        self.ivector = ivector
        self.plda_vector = plda_vector
        self.xvector = xvector
        self.file_id = file_id
        self.speaker_id = speaker_id
        self.job_id = job_id
        self.file = file
        self.speaker = speaker
        self.job = job
        self.phone_intervals = phone_intervals if phone_intervals is not None else []
        self.word_intervals = word_intervals if word_intervals is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def duration(self) -> float:
        """
        Compute and return the duration of the utterance.
        """
        return self.end - self.begin

    @property
    def kaldi_id(self) -> typing.Optional[str]:
        """
        Return a unique Kaldi identifier in the format `speaker_id-id`.
        """
        if self.speaker_id is not None and self.id is not None:
            return f"{self.speaker_id}-{self.id}"
        return None

    def __repr__(self) -> str:
        """
        Return a string representation of the utterance.
        """
        return f"<Utterance in {self.file_name} by {self.speaker_name} from {self.begin} to {self.end}>"

    def phone_intervals_for_workflow(self, workflow_id: int) -> list:
        """
        Extract phone intervals for a given workflow.

        Parameters
        ----------
        workflow_id : int
            The workflow identifier.

        Returns
        -------
        list
            List of phone intervals (converted via as_ctm) matching the workflow.
        """
        return [x.as_ctm() for x in self.phone_intervals if getattr(x, "workflow_id", None) == workflow_id]

    def word_intervals_for_workflow(self, workflow_id: int) -> list:
        """
        Extract word intervals for a given workflow.

        Parameters
        ----------
        workflow_id : int
            The workflow identifier.

        Returns
        -------
        list
            List of word intervals (converted via as_ctm) matching the workflow.
        """
        return [x.as_ctm() for x in self.word_intervals if getattr(x, "workflow_id", None) == workflow_id]

    @property
    def reference_phone_intervals(self) -> list:
        """
        Return phone intervals corresponding to the reference workflow.
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type == WorkflowType.reference
        ]

    @property
    def aligned_phone_intervals(self) -> list:
        """
        Return phone intervals corresponding to alignment workflows.
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type in [WorkflowType.alignment, WorkflowType.online_alignment]
        ]

    @property
    def aligned_word_intervals(self) -> list:
        """
        Return word intervals corresponding to alignment workflows.
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type in [WorkflowType.alignment, WorkflowType.online_alignment]
        ]

    @property
    def transcribed_phone_intervals(self) -> list:
        """
        Return phone intervals corresponding to transcription workflows.
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type in [
                WorkflowType.transcription,
                WorkflowType.per_speaker_transcription,
                WorkflowType.transcript_verification,
            ]
        ]

    @property
    def transcribed_word_intervals(self) -> list:
        """
        Return word intervals corresponding to transcription workflows.
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type in [
                WorkflowType.transcription,
                WorkflowType.per_speaker_transcription,
                WorkflowType.transcript_verification,
            ]
        ]

    @property
    def per_speaker_transcribed_phone_intervals(self) -> list:
        """
        Return phone intervals corresponding to per speaker transcription.
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type == WorkflowType.per_speaker_transcription
        ]

    @property
    def per_speaker_transcribed_word_intervals(self) -> list:
        """
        Return word intervals corresponding to per speaker transcription.
        """
        return [
            x.as_ctm()
            for x in self.word_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type == WorkflowType.per_speaker_transcription
        ]

    @property
    def phone_transcribed_phone_intervals(self) -> list:
        """
        Return phone intervals corresponding to phone transcription workflow.
        """
        return [
            x.as_ctm()
            for x in self.phone_intervals
            if hasattr(x, "workflow") and x.workflow.workflow_type == WorkflowType.phone_transcription
        ]

    @property
    def file_name(self) -> str:
        """
        Return the name of the associated file.
        """
        return self.file.name if self.file is not None else ""

    @property
    def speaker_name(self) -> str:
        """
        Return the name of the associated speaker.
        """
        return self.speaker.name if self.speaker is not None else ""

    def to_data(self):
        """
        Construct an UtteranceData object suitable for multiprocessing.

        Returns
        -------
        UtteranceData
            A data container for the utterance.
        """
        from montreal_forced_aligner.corpus.classes import UtteranceData

        norm_text = self.normalized_text if self.normalized_text is not None else ""
        return UtteranceData(
            self.speaker_name,
            self.file_name,
            self.begin,
            self.end,
            self.channel,
            self.text,
            norm_text.split(),
            set(self.oovs.split()) if self.oovs else set(),
        )

    def to_kalpy(self):
        """
        Construct a KalpyUtterance object for use with Kalpy.

        Returns
        -------
        KalpyUtterance
            A Kalpy-compatible representation of the utterance.
        """
        seg = Segment(self.file.sound_file.sound_file_path, self.begin, self.end, self.channel)
        return KalpyUtterance(seg, self.normalized_text, self.speaker.cmvn, self.speaker.fmllr)

    @classmethod
    def from_data(cls, data, file, speaker, frame_shift: int = None):
        """
        Create an Utterance object from an UtteranceData instance.

        Parameters
        ----------
        data : UtteranceData
            The data for the utterance.
        file : File
            The associated File object.
        speaker : Speaker or int
            The associated Speaker object or its id.
        frame_shift : int, optional
            Frame shift in milliseconds used to calculate the number of frames.

        Returns
        -------
        Utterance
            A new Utterance object.
        """
        if not isinstance(speaker, int):
            speaker = speaker.id
        num_frames = None
        if frame_shift is not None:
            # Compute number of frames using frame_shift (converted from ms to seconds)
            num_frames = int((data.end - data.begin) / round(frame_shift / 1000, 4))
        return cls(
            begin=data.begin,
            end=data.end,
            channel=data.channel,
            oovs=" ".join(sorted(data.oovs)),
            normalized_text=" ".join(data.normalized_text),
            text=data.text,
            num_frames=num_frames,
            file_id=file.id,
            speaker_id=speaker,
        )

class CorpusWorkflow(PolarsModel):
    """
    Polars version of the CorpusWorkflow model for storing information
    about a particular workflow (alignment, transcription, etc).

    Attributes
    ----------
    id : int | None
        Primary key assigned when the record is saved.
    name : str
        Unique name of the workflow.
    workflow_type : any
        Workflow type (typically an enum value or a string).
    working_directory : Path
        Working directory where workflow-related files are stored.
    time_stamp : datetime.datetime
        Timestamp for the workflow run.
    current : bool
        Flag indicating whether this workflow is the current one.
    done : bool
        Flag indicating whether the workflow is complete.
    dirty : bool
        Flag indicating if the workflow data has unsaved changes.
    alignments_collected : bool
        Flag indicating whether alignments have been collected.
    score : float | None
        Score (e.g. log likelihood or another metric) for the workflow run.
    phone_intervals : list
        List of associated PhoneInterval objects.
    word_intervals : list
        List of associated WordInterval objects.
    """

    _table_name = "corpus_workflow"

    def __init__(
        self,
        name: str,
        workflow_type: typing.Any,
        working_directory: typing.Union[str, Path],
        time_stamp: typing.Optional[datetime.datetime] = None,
        current: bool = False,
        done: bool = False,
        dirty: bool = False,
        alignments_collected: bool = False,
        score: typing.Optional[float] = None,
        phone_intervals: list = None,
        word_intervals: list = None,
        **kwargs,
    ):
        self.id = None  # Primary key will be assigned when saved.
        self.name = name
        self.workflow_type = workflow_type
        self.working_directory = (
            working_directory if isinstance(working_directory, Path)
            else Path(working_directory)
        )
        self.time_stamp = time_stamp if time_stamp is not None else datetime.datetime.now()
        self.current = current
        self.done = done
        self.dirty = dirty
        self.alignments_collected = alignments_collected
        self.score = score
        self.phone_intervals = phone_intervals if phone_intervals is not None else []
        self.word_intervals = word_intervals if word_intervals is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def lda_mat_path(self) -> Path:
        """
        Returns the path to the LDA matrix file.
        """
        return self.working_directory.joinpath("lda.mat")

class PhoneInterval(PolarsModel):
    _table_name = "phone_interval"

    def __init__(
        self,
        begin: float,
        end: float,
        phone_goodness: float = None,
        label: str = None,
        phone_id: int = None,
        phone=None,
        word_interval_id: int = None,
        word_interval=None,
        utterance_id: int = None,
        utterance=None,
        workflow_id: int = None,
        workflow=None,
        **kwargs,
    ):
        """
        Initialize a PhoneInterval instance.
        
        Parameters
        ----------
        begin : float
            Beginning timestamp of the interval.
        end : float
            Ending timestamp of the interval.
        phone_goodness : float, optional
            Confidence score or log-likelihood for the phone interval.
        label : str, optional
            (Optional) label from the CTM, used for mapping to a phone.
        phone_id : int, optional
            Identifier for the associated phone.
        phone : object, optional
            The associated phone instance or data.
        word_interval_id : int, optional
            Identifier for the related word interval.
        word_interval : object, optional
            The associated word interval instance or data.
        utterance_id : int, optional
            Identifier for the associated utterance.
        utterance : object, optional
            The associated utterance instance or data.
        workflow_id : int, optional
            Identifier for the workflow that generated the interval.
        workflow : object, optional
            The associated workflow instance or data.
        kwargs :
            Any additional keyword arguments.
        """
        self.id = None
        self.begin = begin
        self.end = end
        self.phone_goodness = phone_goodness
        self.label = label
        self.phone_id = phone_id
        self.phone = phone
        self.word_interval_id = word_interval_id
        self.word_interval = word_interval
        self.utterance_id = utterance_id
        self.utterance = utterance
        self.workflow_id = workflow_id
        self.workflow = workflow
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def duration(self) -> float:
        """
        Computed duration of the interval.
        """
        return self.end - self.begin

    def __repr__(self):
        """
        Return a human-readable representation of the PhoneInterval.
        """
        phone_label = (
            self.phone.kaldi_label if (self.phone and hasattr(self.phone, "kaldi_label")) else "N/A"
        )
        workflow_type = (
            self.workflow.workflow_type
            if (self.workflow and hasattr(self.workflow, "workflow_type"))
            else "N/A"
        )
        utt_id = self.utterance_id if self.utterance_id is not None else "N/A"
        return (
            f"<PhoneInterval {phone_label} ({workflow_type}) from {self.begin}-{self.end} "
            f"for utterance {utt_id}>"
        )

    @classmethod
    def from_ctm(cls, interval, utterance, workflow_id: int):
        """
        Construct a PhoneInterval from a CtmInterval object.
        
        Parameters
        ----------
        interval : CtmInterval
            Object containing data for the phone interval (e.g., attributes begin, end, label).
        utterance : Utterance
            Utterance instance that the phone interval belongs to.
        workflow_id : int
            Identifier for the workflow that generated the phone interval.
        
        Returns
        -------
        PhoneInterval
            A new PhoneInterval instance.
        """
        return cls(
            begin=interval.begin,
            end=interval.end,
            label=interval.label,
            utterance=utterance,
            workflow_id=workflow_id,
        )

    def as_ctm(self, CtmInterval):
        """
        Generate a CtmInterval object from the PhoneInterval.

        Parameters
        ----------
        CtmInterval : type
            A callable (or class) that takes (begin, end, phone, confidence) and returns a CtmInterval.
        
        Returns
        -------
        CtmInterval
            A new CTM interval constructed from this phone interval.
        """
        # It is assumed that self.phone is an object with a 'phone' attribute.
        return CtmInterval(self.begin, self.end, self.phone.phone, confidence=self.phone_goodness)
    
class WordInterval(PolarsModel):
    _table_name = "word_interval"

    def __init__(
        self,
        begin: float,
        end: float,
        label: str = None,
        utterance_id: int = None,
        utterance=None,
        word_id: int = None,
        word=None,
        pronunciation_id: int = None,
        pronunciation=None,
        workflow_id: int = None,
        workflow=None,
        phone_intervals: list = None,
        **kwargs,
    ):
        """
        Initialize a WordInterval instance.
        
        Parameters
        ----------
        begin : float
            Beginning timestamp of the interval.
        end : float
            Ending timestamp of the interval.
        label : str, optional
            Optional label from the CTM, used for mapping the word.
        utterance_id : int, optional
            Identifier for the associated utterance.
        utterance : object, optional
            The associated utterance instance.
        word_id : int, optional
            Identifier for the word.
        word : object, optional
            The associated word instance.
        pronunciation_id : int, optional
            Identifier for the pronunciation.
        pronunciation : object, optional
            The associated pronunciation instance.
        workflow_id : int, optional
            Identifier for the associated workflow.
        workflow : object, optional
            The associated workflow instance.
        phone_intervals : list, optional
            List of associated phone intervals.
        kwargs :
            Any additional keyword arguments.
        """
        self.id = None
        self.begin = begin
        self.end = end
        self.label = label
        self.utterance_id = utterance_id
        self.utterance = utterance
        self.word_id = word_id
        self.word = word
        self.pronunciation_id = pronunciation_id
        self.pronunciation = pronunciation
        self.workflow_id = workflow_id
        self.workflow = workflow
        self.phone_intervals = phone_intervals if phone_intervals is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def duration(self) -> float:
        """
        Computed duration of the interval.
        """
        return self.end - self.begin

    @classmethod
    def from_ctm(cls, interval, utterance, workflow_id: int):
        """
        Construct a WordInterval from a CtmInterval object.
        
        Parameters
        ----------
        interval : CtmInterval
            An object containing data for the word interval (e.g., attributes begin, end, label).
        utterance : object
            The utterance instance that the word interval belongs to.
        workflow_id : int
            Identifier for the workflow that generated the interval.
        
        Returns
        -------
        WordInterval
            A new WordInterval instance.
        """
        return cls(
            begin=interval.begin,
            end=interval.end,
            label=interval.label,
            utterance=utterance,
            workflow_id=workflow_id,
        )

    def as_ctm(self, CtmInterval):
        """
        Generate a CtmInterval object from the WordInterval.
        
        Parameters
        ----------
        CtmInterval : type
            A callable (or class) that takes (begin, end, word) and returns a CtmInterval.
        
        Returns
        -------
        CtmInterval
            A new CtmInterval object constructed from this WordInterval.
        """
        # It is assumed that self.word is an object having an attribute 'word'
        return CtmInterval(self.begin, self.end, self.word.word)
    
class Job(PolarsModel):
    _table_name = "job"

    def __init__(
        self,
        corpus_id: int = None,
        corpus=None,
        utterances: list = None,
        symbols: list = None,
        words: list = None,
        dictionaries: list = None,
        **kwargs,
    ):
        """
        Initialize a Job instance.

        Parameters
        ----------
        corpus_id : int, optional
            Identifier for the associated corpus.
        corpus : object, optional
            The corpus associated with the job.
        utterances : list, optional
            List of utterance objects associated with the job.
        symbols : list, optional
            Symbols (M2M2Job objects) associated with the job.
        words : list, optional
            Words (Word2Job objects) associated with the job.
        dictionaries : list, optional
            Dictionary objects associated with the job.
        kwargs :
            Additional keyword arguments.
        """
        self.id = None
        self.corpus_id = corpus_id
        self.corpus = corpus
        self.utterances = utterances if utterances is not None else []
        self.symbols = symbols if symbols is not None else []
        self.words = words if words is not None else []
        self.dictionaries = dictionaries if dictionaries is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"<Job {self.id}>"

    @property
    def has_dictionaries(self) -> bool:
        return len(self.dictionaries) > 0

    @property
    def training_dictionaries(self) -> typing.List:
        """
        Return a list of training dictionaries based on the corpus' current subset.
        """
        if self.corpus.current_subset == 0:
            return self.dictionaries
        if self.corpus.current_subset <= 25000:
            return [d for d in self.dictionaries if d.name not in {"default", "nonnative"}]
        return [d for d in self.dictionaries if d.name not in {"default"}]

    @property
    def dictionary_ids(self) -> typing.List[int]:
        return [d.id for d in self.dictionaries]

    def construct_path(
        self, directory: Path, identifier: str, extension: str, dictionary_id: int = None
    ) -> Path:
        """
        Helper function for constructing dictionary-dependent paths.

        Parameters
        ----------
        directory : Path
            Root directory.
        identifier : str
            An identifier for the filename (e.g., "trans", "wav").
        extension : str
            The file extension (e.g., "scp", "ark").
        dictionary_id : int, optional
            If provided, include the dictionary id in the file name.

        Returns
        -------
        Path
            The constructed file path.
        """
        if dictionary_id is None:
            return directory.joinpath(f"{identifier}.{self.id}.{extension}")
        return directory.joinpath(f"{identifier}.{dictionary_id}.{self.id}.{extension}")

    def construct_path_dictionary(
        self, directory: Path, identifier: str, extension: str
    ) -> typing.Dict[int, Path]:
        """
        Constructs a mapping from dictionary id to its constructed file path.
        """
        return {d.id: self.construct_path(directory, identifier, extension, d.id) for d in self.dictionaries}

    def construct_dictionary_dependent_paths(
        self, directory: Path, identifier: str, extension: str
    ) -> typing.Dict[int, Path]:
        """
        Constructs paths that depend solely on the dictionary id.
        """
        output = {}
        for d in self.dictionaries:
            output[d.id] = directory.joinpath(f"{identifier}.{d.id}.{extension}")
        return output

    @property
    def wav_scp_path(self) -> Path:
        return self.construct_path(self.corpus.split_directory, "wav", "scp")

    @property
    def segments_scp_path(self) -> Path:
        return self.construct_path(self.corpus.split_directory, "segments", "scp")

    @property
    def utt2spk_scp_path(self) -> Path:
        return self.construct_path(self.corpus.split_directory, "utt2spk", "scp")

    @property
    def feats_scp_path(self) -> Path:
        return self.construct_path(self.corpus.split_directory, "feats", "scp")

    @property
    def feats_ark_path(self) -> Path:
        return self.construct_path(self.corpus.split_directory, "feats", "ark")

    @property
    def per_dictionary_feats_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "feats", "scp", d.id)
        return paths

    @property
    def per_dictionary_utt2spk_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "utt2spk", "scp", d.id)
        return paths

    @property
    def per_dictionary_spk2utt_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "spk2utt", "scp", d.id)
        return paths

    @property
    def per_dictionary_cmvn_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "cmvn", "scp", d.id)
        return paths

    @property
    def per_dictionary_trans_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "trans", "scp", d.id)
        return paths

    @property
    def per_dictionary_text_int_scp_paths(self) -> typing.Dict[int, Path]:
        paths = {}
        for d in self.dictionaries:
            paths[d.id] = self.construct_path(self.corpus.current_subset_directory, "text", "int.scp", d.id)
        return paths

    def construct_feature_archive(
        self, working_directory: Path, dictionary_id: typing.Optional[int] = None, **kwargs
    ):
        """
        Construct a FeatureArchive object from various feature file paths.
        
        Parameters
        ----------
        working_directory : Path
            The base working directory containing the "lda.mat" file.
        dictionary_id : int, optional
            Specific dictionary id for which to build archive paths.
        kwargs :
            Additional keyword arguments for FeatureArchive.
        
        Returns
        -------
        FeatureArchive
            A new FeatureArchive object constructed with the gathered paths.
        """
        fmllr_path = self.construct_path(self.corpus.current_subset_directory, "trans", "scp", dictionary_id)
        if not fmllr_path.exists():
            fmllr_path = None
            utt2spk = None
        else:
            utt2spk_path = self.construct_path(self.corpus.current_subset_directory, "utt2spk", "scp", dictionary_id)
            utt2spk = KaldiMapping()
            utt2spk.load(utt2spk_path)
        lda_mat_path = working_directory.joinpath("lda.mat")
        if not lda_mat_path.exists():
            lda_mat_path = None
        feat_path = self.construct_path(self.corpus.current_subset_directory, "feats", "scp", dictionary_id)
        vad_path = self.construct_path(self.corpus.current_subset_directory, "vad", "scp", dictionary_id)
        if not vad_path.exists():
            vad_path = None
        feature_archive = FeatureArchive(
            feat_path,
            utt2spk=utt2spk,
            lda_mat_file_name=lda_mat_path,
            transform_file_name=fmllr_path,
            vad_file_name=vad_path,
            deltas=True,
            **kwargs,
        )
        return feature_archive
    
class M2MSymbol(PolarsModel):
    _table_name = "m2m_symbol"

    def __init__(
        self,
        symbol: str,
        total_order: int,
        max_order: int,
        grapheme_order: int,
        phone_order: int,
        weight: float,
        jobs: list = None,
        **kwargs,
    ):
        """
        Initialize an M2MSymbol instance.
        
        Parameters
        ----------
        symbol : str
            The symbol.
        total_order : int
            Summed order of graphemes and phones.
        max_order : int
            Maximum order between graphemes and phones.
        grapheme_order : int
            Grapheme order.
        phone_order : int
            Phone order.
        weight : float
            Weight of arcs.
        jobs : list, optional
            List of associated M2M2Job objects.
        kwargs :
            Any additional keyword arguments.
        """
        self.id = None
        self.symbol = symbol
        self.total_order = total_order
        self.max_order = max_order
        self.grapheme_order = grapheme_order
        self.phone_order = phone_order
        self.weight = weight
        self.jobs = jobs if jobs is not None else []
        for key, value in kwargs.items():
            setattr(self, key, value)


class M2M2Job(PolarsModel):
    _table_name = "m2m_job"

    def __init__(self, m2m_id: int, job_id: int, m2m_symbol=None, job=None, **kwargs):
        """
        Initialize an M2M2Job instance.
        
        Parameters
        ----------
        m2m_id : int
            Identifier for the associated M2MSymbol.
        job_id : int
            Identifier for the associated Job.
        m2m_symbol : object, optional
            The associated M2MSymbol object.
        job : object, optional
            The associated Job object.
        kwargs :
            Additional keyword arguments.
        """
        self.m2m_id = m2m_id
        self.job_id = job_id
        self.m2m_symbol = m2m_symbol
        self.job = job
        for key, value in kwargs.items():
            setattr(self, key, value)


class Word2Job(PolarsModel):
    _table_name = "word_job"

    def __init__(
        self,
        word_id: int,
        job_id: int,
        training: bool,
        word=None,
        job=None,
        **kwargs,
    ):
        """
        Initialize a Word2Job instance.
        
        Parameters
        ----------
        word_id : int
            Identifier for the associated Word.
        job_id : int
            Identifier for the associated Job.
        training : bool
            A flag indicating the training status.
        word : object, optional
            The associated Word object.
        job : object, optional
            The associated Job object.
        kwargs :
            Additional keyword arguments.
        """
        self.word_id = word_id
        self.job_id = job_id
        self.training = training
        self.word = word
        self.job = job
        for key, value in kwargs.items():
            setattr(self, key, value)