"""Classes for calculating alignments online"""
from __future__ import annotations

import typing

import sqlalchemy.orm
from _kalpy.matrix import DoubleMatrix, FloatMatrix
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.fstext.lexicon import Pronunciation as KalpyPronunciation
from kalpy.gmm.align import GmmAligner
from kalpy.gmm.data import HierarchicalCtm
from kalpy.utterance import Utterance as KalpyUtterance

from montreal_forced_aligner.data import Language, WordType
from montreal_forced_aligner.db_polars import (
    Phone,
    PhoneInterval,
    Pronunciation,
    Utterance,
    Word,
    WordInterval,
)
from montreal_forced_aligner.exceptions import AlignerError
from montreal_forced_aligner.models import AcousticModel, G2PModel


def align_utterance_online(
    acoustic_model: AcousticModel,
    utterance: KalpyUtterance,
    lexicon_compiler: LexiconCompiler,
    tokenizer=None,
    g2p_model: G2PModel = None,
    cmvn: DoubleMatrix = None,
    fmllr_trans: FloatMatrix = None,
    beam: int = 10,
    retry_beam: int = 40,
    transition_scale: float = 1.0,
    acoustic_scale: float = 0.1,
    self_loop_scale: float = 0.1,
    boost_silence: float = 1.0,
) -> HierarchicalCtm:
    text = utterance.transcript
    rewriter = None
    if g2p_model is not None:
        rewriter = g2p_model.rewriter
    if tokenizer is not None:
        if acoustic_model.language is Language.unknown:
            text, _, oovs = tokenizer(text)
            if rewriter is not None:
                for w in oovs:
                    if not lexicon_compiler.word_table.member(w):
                        pron = rewriter(w)
                        if pron:
                            lexicon_compiler.add_pronunciation(
                                KalpyPronunciation(w, pron[0], None, None, None, None, None)
                            )

        else:
            text, pronunciation_form = tokenizer(text)
            if not pronunciation_form:
                pronunciation_form = text
            g2p_cache = {}
            if rewriter is not None:
                for norm_w, w in zip(text.split(), pronunciation_form.split()):
                    if w not in g2p_cache:
                        pron = rewriter(w)
                        if not pron:
                            continue
                        g2p_cache[w] = pron[0]
                    if w in g2p_cache and not lexicon_compiler.word_table.member(norm_w):
                        lexicon_compiler.add_pronunciation(
                            KalpyPronunciation(norm_w, g2p_cache[w], None, None, None, None, None)
                        )

    graph_compiler = TrainingGraphCompiler(
        acoustic_model.alignment_model_path,
        acoustic_model.tree_path,
        lexicon_compiler,
    )
    if utterance.mfccs is None:
        utterance.generate_mfccs(acoustic_model.mfcc_computer)
        if acoustic_model.uses_cmvn:
            if cmvn is None:
                cmvn_computer = CmvnComputer()
                cmvn = cmvn_computer.compute_cmvn_from_features([utterance.mfccs])
            utterance.apply_cmvn(cmvn)
    feats = utterance.generate_features(
        acoustic_model.mfcc_computer,
        acoustic_model.pitch_computer,
        lda_mat=acoustic_model.lda_mat,
        fmllr_trans=fmllr_trans,
    )

    fst = graph_compiler.compile_fst(text)
    aligner = GmmAligner(
        acoustic_model.alignment_model_path if fmllr_trans is None else acoustic_model.model_path,
        beam=beam,
        retry_beam=retry_beam,
        transition_scale=transition_scale,
        acoustic_scale=acoustic_scale,
        self_loop_scale=self_loop_scale,
    )
    if boost_silence != 1.0:
        aligner.boost_silence(boost_silence, lexicon_compiler.silence_symbols)
    alignment = aligner.align_utterance(fst, feats)
    if alignment is None:
        raise AlignerError(
            f"Could not align the file with the current beam size ({aligner.beam}, "
            "please try increasing the beam size via `--beam X`"
        )
    phone_intervals = alignment.generate_ctm(
        aligner.transition_model,
        lexicon_compiler.phone_table,
        acoustic_model.mfcc_computer.frame_shift,
    )
    ctm = lexicon_compiler.phones_to_pronunciations(
        alignment.words, phone_intervals, transcription=False, text=utterance.transcript
    )
    ctm.likelihood = alignment.likelihood
    ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
    return ctm


# def update_utterance_intervals(
#     session: sqlalchemy.orm.Session,
#     utterance: typing.Union[int, Utterance],
#     workflow_id: int,
#     ctm: HierarchicalCtm,
# ):
#     if isinstance(utterance, int):
#         utterance = session.get(Utterance, utterance)
#     max_phone_interval_id = session.query(sqlalchemy.func.max(PhoneInterval.id)).scalar()
#     if max_phone_interval_id is None:
#         max_phone_interval_id = 0
#     max_word_interval_id = session.query(sqlalchemy.func.max(WordInterval.id)).scalar()
#     if max_word_interval_id is None:
#         max_word_interval_id = 0
#     mapping_id = session.query(sqlalchemy.func.max(Word.mapping_id)).scalar()
#     if mapping_id is None:
#         mapping_id = -1
#     mapping_id += 1
#     word_index = get_next_primary_key(session, Word)
#     new_phone_interval_mappings = []
#     new_word_interval_mappings = []
#     words = (
#         session.query(Word.word, Word.id)
#         .filter(Word.dictionary_id == utterance.speaker.dictionary_id)
#         .filter(Word.word.in_(utterance.normalized_text.split()))
#     )
#     phone_to_phone_id = {}
#     word_mapping = {}
#     pronunciation_mapping = {}
#     ds = session.query(Phone.id, Phone.mapping_id).all()
#     for p_id, mapping_id in ds:
#         phone_to_phone_id[mapping_id] = p_id
#     new_words = []
#     for w, w_id in words:
#         word_mapping[w] = w_id
#     pronunciations = (
#         session.query(Word.word, Pronunciation.pronunciation, Pronunciation.id)
#         .join(Pronunciation.word)
#         .filter(Word.dictionary_id == utterance.speaker.dictionary_id)
#         .filter(Word.word.in_(utterance.normalized_text.split()))
#     )
#     for w, pron, p_id in pronunciations:
#         pronunciation_mapping[(w, pron)] = p_id
#     for word_interval in ctm.word_intervals:
#         if word_interval.label not in word_mapping:
#             new_words.append(
#                 {
#                     "id": word_index,
#                     "mapping_id": mapping_id,
#                     "word": word_interval.label,
#                     "dictionary_id": 1,
#                     "word_type": WordType.oov,
#                 }
#             )
#             word_mapping[word_interval.label] = word_index
#             word_id = word_index
#         else:
#             word_id = word_mapping[word_interval.label]
#         max_word_interval_id += 1
#         pronunciation_id = pronunciation_mapping.get(
#             (word_interval.label, word_interval.pronunciation), None
#         )

#         new_word_interval_mappings.append(
#             {
#                 "id": max_word_interval_id,
#                 "begin": word_interval.begin,
#                 "end": word_interval.end,
#                 "word_id": word_id,
#                 "pronunciation_id": pronunciation_id,
#                 "utterance_id": utterance.id,
#                 "workflow_id": workflow_id,
#             }
#         )
#         for interval in word_interval.phones:
#             max_phone_interval_id += 1
#             new_phone_interval_mappings.append(
#                 {
#                     "id": max_phone_interval_id,
#                     "begin": interval.begin,
#                     "end": interval.end,
#                     "phone_id": phone_to_phone_id[interval.symbol],
#                     "utterance_id": utterance.id,
#                     "workflow_id": workflow_id,
#                     "word_interval_id": max_word_interval_id,
#                     "phone_goodness": interval.confidence if interval.confidence else 0.0,
#                 }
#             )
#     session.query(Utterance).filter(Utterance.id == utterance.id).update(
#         {Utterance.alignment_log_likelihood: ctm.likelihood}
#     )
#     session.query(PhoneInterval).filter(PhoneInterval.utterance_id == utterance.id).filter(
#         PhoneInterval.workflow_id == workflow_id
#     ).delete(synchronize_session=False)
#     session.flush()
#     session.query(WordInterval).filter(WordInterval.utterance_id == utterance.id).filter(
#         WordInterval.workflow_id == workflow_id
#     ).delete(synchronize_session=False)
#     session.flush()
#     if new_words:
#         session.bulk_insert_mappings(Word, new_words, return_defaults=False, render_nulls=True)
#         session.flush()
#     if new_word_interval_mappings:
#         session.bulk_insert_mappings(
#             WordInterval, new_word_interval_mappings, return_defaults=False, render_nulls=True
#         )
#         session.flush()
#         session.bulk_insert_mappings(
#             PhoneInterval,
#             new_phone_interval_mappings,
#             return_defaults=False,
#             render_nulls=True,
#         )
#     session.commit()
def update_utterance_intervals(
    db,  # PolarsDB instance instead of a SQLAlchemy session
    utterance: typing.Union[int, Utterance],
    workflow_id: int,
    ctm: HierarchicalCtm,
):
    """
    Update the utterance phone and word interval tables using the Polars in‑memory database.
    
    Parameters
    ----------
    db : PolarsDB
        The in‑memory Polars database instance.
    utterance : int or Utterance
        The utterance or its ID.
    workflow_id : int
        Current workflow identifier.
    ctm : HierarchicalCtm
        The CTM (alignment) object.
    """

    # If utterance is passed as an integer, look it up in the "utterance" table.
    if isinstance(utterance, int):
        utt_df = db.get_table("utterance")
        row = utt_df.filter(pl.col("id") == utterance)
        if row.height == 0:
            raise ValueError(f"Utterance with id {utterance} not found")
        utterance = row.row(0, named=True)

    # Compute current max primary keys from phone_interval and word_interval tables.
    phone_interval_df = db.get_table("phone_interval")
    if phone_interval_df.is_empty() or "id" not in phone_interval_df.columns:
        max_phone_interval_id = 0
    else:
        max_phone_interval_id = phone_interval_df["id"].max()

    word_interval_df = db.get_table("word_interval")
    if word_interval_df.is_empty() or "id" not in word_interval_df.columns:
        max_word_interval_id = 0
    else:
        max_word_interval_id = word_interval_df["id"].max()

    # Get the current maximum mapping_id from the word table.
    word_df = db.get_table("word")
    if word_df.is_empty() or "mapping_id" not in word_df.columns:
        mapping_id = -1
    else:
        mapping_id = word_df["mapping_id"].max() or -1
    mapping_id += 1

    # Get the next primary key for the "word" table.
    word_index = db.get_next_primary_key("word")

    new_phone_interval_mappings = []
    new_word_interval_mappings = []

    # Build a mapping from phone mapping_id to phone id.
    phone_table = db.get_table("phone")
    phone_to_phone_id = {}
    for row in phone_table.iter_rows(named=True):
        phone_to_phone_id[row["mapping_id"]] = row["id"]

    # Build a mapping for existing words.
    word_mapping = {}
    utterance_words = utterance.normalized_text.split()
    words_df = db.get_table("word").filter(
        (pl.col("dictionary_id") == utterance.speaker.dictionary_id)
        & (pl.col("word").is_in(utterance_words))
    )
    for row in words_df.iter_rows(named=True):
        word_mapping[row["word"]] = row["id"]

    # Build a pronunciation mapping by joining the word and pronunciation tables.
    pronunciation_mapping = {}
    pron_df = db.get_table("pronunciation")
    word_for_join = db.get_table("word")
    # Perform an inner join to get the associated word (assumes "word_id" links pronunciation to word id)
    joined = pron_df.join(word_for_join, left_on="word_id", right_on="id", how="inner")
    joined = joined.filter(
        (pl.col("dictionary_id") == utterance.speaker.dictionary_id)
        & (pl.col("word").is_in(utterance_words))
    )
    for row in joined.iter_rows(named=True):
        # Map the tuple (word, pronunciation) to the pronunciation's id.
        pronunciation_mapping[(row["word"], row["pronunciation"])] = row["id"]

    new_words = []
    # Process each word interval from the CTM.
    for word_interval in ctm.word_intervals:
        if word_interval.label not in word_mapping:
            new_words.append(
                {
                    "id": word_index,
                    "mapping_id": mapping_id,
                    "word": word_interval.label,
                    "dictionary_id": 1,  # Adjust if your dictionary ID is determined dynamically
                    "word_type": WordType.oov,
                }
            )
            word_mapping[word_interval.label] = word_index
            word_id = word_index
            word_index += 1
        else:
            word_id = word_mapping[word_interval.label]
        max_word_interval_id += 1
        pronunciation_id = pronunciation_mapping.get(
            (word_interval.label, word_interval.pronunciation), None
        )
        new_word_interval_mappings.append(
            {
                "id": max_word_interval_id,
                "begin": word_interval.begin,
                "end": word_interval.end,
                "word_id": word_id,
                "pronunciation_id": pronunciation_id,
                "utterance_id": utterance.id,
                "workflow_id": workflow_id,
            }
        )
        # For every phone interval in this word interval...
        for interval in word_interval.phones:
            max_phone_interval_id += 1
            new_phone_interval_mappings.append(
                {
                    "id": max_phone_interval_id,
                    "begin": interval.begin,
                    "end": interval.end,
                    "phone_id": phone_to_phone_id.get(interval.symbol, None),
                    "utterance_id": utterance.id,
                    "workflow_id": workflow_id,
                    "word_interval_id": max_word_interval_id,
                    "phone_goodness": interval.confidence if interval.confidence else 0.0,
                }
            )

    # Update the alignment log likelihood for the utterance.
    db.bulk_update("utterance", [{"id": utterance.id, "alignment_log_likelihood": ctm.likelihood}])

    # Delete any existing phone intervals for this utterance and workflow.
    current_phone_intervals = db.get_table("phone_interval")
    updated_phone_intervals = current_phone_intervals.filter(
        ~((pl.col("utterance_id") == utterance.id) & (pl.col("workflow_id") == workflow_id))
    )
    db.replace_table("phone_interval", updated_phone_intervals)

    # Delete any existing word intervals for this utterance and workflow.
    current_word_intervals = db.get_table("word_interval")
    updated_word_intervals = current_word_intervals.filter(
        ~((pl.col("utterance_id") == utterance.id) & (pl.col("workflow_id") == workflow_id))
    )
    db.replace_table("word_interval", updated_word_intervals)

    # Insert any new words that were not yet in the dictionary.
    if new_words:
        for row in new_words:
            db.add_row("word", row)

    # Insert the new word interval mappings.
    if new_word_interval_mappings:
        for row in new_word_interval_mappings:
            db.add_row("word_interval", row)

    # Insert the new phone interval mappings.
    if new_phone_interval_mappings:
        for row in new_phone_interval_mappings:
            db.add_row("phone_interval", row)

    # No explicit commit is needed for the in‑memory Polars DB.