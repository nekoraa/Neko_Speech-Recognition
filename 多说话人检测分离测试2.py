from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-whamr16k", savedir='pretrained_models/sepformer-whamr16k')

# for custom file, change path
est_sources = model.separate_file(path='speechbrain/sepformer-whamr16k/test_mixture16k.wav')

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 16000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 16000)
