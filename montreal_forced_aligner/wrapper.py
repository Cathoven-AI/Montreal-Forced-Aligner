from pathlib import Path
from montreal_forced_aligner.alignment.preloaded_pretrained_aligner import PreloadedPretrainedAligner
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
from montreal_forced_aligner.models import AcousticModel
import polars as pl
import soundfile
import warnings
import re
import tempfile
import multiprocessing as mp
import pyfoal

class MFA:
    def __init__(self, model='english_us_arpa', num_jobs=4, update_models: bool = False):
        self.num_jobs = num_jobs
        self.model = model

        # Initialize the ModelManager with the update flag.
        from montreal_forced_aligner.models import ModelManager
        self.model_manager = ModelManager(update=update_models)

        # Check for dictionary and acoustic model in the local cache;
        # if not present (or update is requested) then download them.
        if not self.model_manager.has_local_model("dictionary", model):
            self.model_manager.download_model("dictionary", model, update=update_models)
        if not self.model_manager.has_local_model("acoustic", model):
            self.model_manager.download_model("acoustic", model, update=update_models)

        # Load the dictionary and acoustic model from pretrained_models directory.
        self.dictionary = load_dictionary(model)
        self.acoustic_model = load_acoustic_model(model)
        
        # Initialize the aligner with the preloaded objects.
        # self.aligner = PreloadedPretrainedAligner(
        #     dictionary=self.dictionary,
        #     acoustic_model=self.acoustic_model,
        #     num_jobs=self.num_jobs
        # )
        self.aligner = PretrainedAligner(
            dictionary=self.dictionary,
            acoustic_model=self.acoustic_model,
            num_jobs=self.num_jobs
        )

    def align(self, data, output_dir, sample_rate=pyfoal.SAMPLE_RATE):
        """
        Run the aligner on the specified corpus directory.
        It is assumed that the corpus directory contains the necessary data files.
        """
        

        with tempfile.TemporaryDirectory() as directory:
            with pyfoal.chdir(directory):
                # Create files directly in speaker directories
                with mp.Pool(self.num_jobs) as pool:
                    args = [(directory, x, sample_rate) for x in data]
                    results = pool.starmap(create_files_worker, args)
                
                # Align
                self.aligner.align(corpus_directory=Path(directory))
                
                # Replace all bracketed content with their original content
                for text, textgrid_file in results:

                    brackets = re.findall(r'[\(\{\[].+?[\)\}\]]', text)
                    
                    try:
                        with open(textgrid_file, 'r') as f:
                            textgrid_content = f.read()
                        
                        for bracket in brackets:
                            textgrid_content = textgrid_content.replace('[bracketed]', bracket, 1)
                        
                        with open(textgrid_file, 'w') as f:
                            f.write(textgrid_content)
                            
                    except FileNotFoundError:
                        warnings.warn(f'Could not find TextGrid file: {textgrid_file}')


                #     # The alignment can fail. This typically indicates that the
                #     # transcript and audio do not match. We skip these files.
                #     try:
                #         pypar.Alignment(textgrid_file).save(output_file)
                #     except FileNotFoundError:
                #         warnings.warn('MFA failed to align at least one file')

                # Export alignments
                self.aligner.export_files(output_dir)

def load_dictionary(model_name: str):
    """
    Load a pretrained dictionary into a Polars DataFrame.
    
    The dictionary file is assumed to be stored at:
      [base_dir]/{model_name}.txt,
    where the base_dir is obtained from the DictionaryModel class.
    """
    from montreal_forced_aligner.models import DictionaryModel
    # Get the base directory from the model manager instead of hardcoding it.
    base_dir = DictionaryModel.pretrained_directory()
    dict_path = base_dir / f"{model_name}.dict"
    
    records = []
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue  # skip malformed lines
            word = tokens[0]
            i = 1
            while i < len(tokens):
                try:
                    float(tokens[i])
                    i += 1
                except ValueError:
                    break
            pronunciation = " ".join(tokens[i:])
            records.append({"word": word, "pronunciation": pronunciation})
    
    df = pl.DataFrame(records)
    return df

def load_acoustic_model(model_name: str):
    """
    Load the AcousticModel. Here we use the AcousticModel class
    from the package directly. This assumes the model is stored at:
      [base_dir]/{model_name},
    where the base_dir is obtained from the AcousticModel class.
    """
    from montreal_forced_aligner.models import AcousticModel
    # Get the correct base directory from the model manager rather than hardcoding.
    base_dir = AcousticModel.pretrained_directory()
    model_path = base_dir / f"{model_name}.zip"
    model = AcousticModel(model_path)
    return model

def create_files_worker(directory, data_item, sample_rate):
    """Worker function to create files and initialize aligner per process"""

    speaker_directory = Path(directory) / data_item['speaker_id']
    speaker_directory.mkdir(exist_ok=True, parents=True)
    
    # Create text file
    text_file = speaker_directory / f"{data_item['file_id']}.txt"
    with open(text_file, 'w') as f:
        f.write(data_item['text'])
        
    # Create audio file
    audio_file = speaker_directory / f"{data_item['file_id']}.wav"
    audio = pyfoal.load.audio(data_item['audio_path']).squeeze().numpy()
    soundfile.write(str(audio_file), audio, sample_rate)
    
    textgrid_file = speaker_directory / f'{data_item["file_id"]}.TextGrid'
    
    return (
        data_item['text'],
        textgrid_file.resolve()
    )