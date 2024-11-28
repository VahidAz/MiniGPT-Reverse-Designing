# MiniGPT-Reverse-Designing

This repository is the impelementation for the <font size='5'>**MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4**</font> <a href='https://arxiv.org/abs/2406.00971'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<p align="center">
      <img src=figs/model.png width=500, height=500>
</p>
<p align="center">
      <img src=figs/output_sample.png width=500, height=500>
</p>

Reverse designing,  which could be defined as a complex vision-language task, aims to predict the edits and their parameters, given a source image, an edited version, and an optional high-level textual edit description. This task requires VLMs to comprehend the interplay between the source image, the edited version, and the optional textual context simultaneously, going beyond traditional vision-language tasks. In this paper, we extend and fine-tune MiniGPT-4 for the reverse designing task. Our experiments demonstrate the extensibility of off-the-shelf VLMs, specifically  MiniGPT-4, for more complex tasks such as reverse designing.

### Dataset
The [I-MAD dataset](https://gamma.umd.edu/researchdirections/affectivecomputing/tame_rd/) has been used for this work.

For dataset preparation, please run the following scripts sequentially:
- `data/download_IMAD_dense.ipynb`
- `data/dataset_prepration.ipynb`

### LLAMA2
The large language model (LLM) used is [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).  
Use `download_llama2.ipynb` to download the model. 

Set the LLM path in the [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15 to 'Llama-2-7b-chat-hf'.

### Training
The list of required Python packages is available in `requirements.txt`. Ensure you have these installed or install them before training.

[Download MiniGPT-4 checkpoint](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) to the 'stage_2_checkpoint'.

```
python train.py --cfg-path train_configs/minigpt_llama2_stage2_finetune_IMAD.yaml
```
the model will be trained based on the experiment No. 2 configuration (Fine-tuning MiniGPT-4 with MSE as Auxiliary Loss), to replicate other experiments, please comment/uncomment lines in [minigpt_base.py](minigpt4/models/minigpt_base.py#L436-L471)

### Evaluation
Please add the path to your desired checkpoint at [minigpt_llama2_eval_IMAD.yaml](eval_configs/minigpt_llama2_eval_IMAD.yaml#L10)
```
python eval_IMAD.py --cfg-path eval_configs/minigpt_llama2_eval_IMAD.yaml  --gpu-id 0
```

### Experiment No. 2; best checkpoint
[Download](https://drive.google.com/file/d/1P-JrX5_iBvTJH7a933H9_T5_Q7mbAFta/view?usp=sharing)

## Acknowledgement
+ [I-MAD](https://gamma.umd.edu/researchdirections/affectivecomputing/tame_rd/) The dataset used in this work.
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) This repository is built upon MiniGPT-4.
+ [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) The Llama 2 language model.

## Citation
Please consider citing the following paper if you use our method in your research:
```bibtex
@misc{azizi2024minigptreversedesigning,
      title={MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4}, 
      author={Vahid Azizi and Fatemeh Koochaki},
      year={2024},
      eprint={2406.00971},
      archivePrefix={arXiv}
    }
```

## License
This repository is licensed under the [BSD 3-Clause License](LICENSE).
We used [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) as our base code with BSD 3-Clause License [here](LICENSE.md).
