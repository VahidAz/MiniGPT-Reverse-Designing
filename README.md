# MiniGPT-Reverse-Designing

<font size='5'>**MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4**</font>

Vahid Azizi, Fatemeh Koochaki

<a href='https://arxiv.org/abs/2406.00971'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


### Training
```
python train.py --cfg-path train_configs/minigpt_llama2_stage2_finetune_IMAD.yaml
```

### Evaluation
```
python eval_IMAD.py --cfg-path eval_configs/minigpt_llama2_eval_IMAD.yaml  --gpu-id 0
```

### Experiment No. 2; best checkpoint
[Download](https://drive.google.com/file/d/1P-JrX5_iBvTJH7a933H9_T5_Q7mbAFta/view?usp=sharing)

## Acknowledgement
+ [I-MAD](https://gamma.umd.edu/researchdirections/affectivecomputing/tame_rd/) The dataset used in this work.
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) This repository is built upon MiniGPT-4.
+ [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) The Llama 2 language model.

If you're using MiniGPT-Reverse-Designing, please cite using this BibTeX:
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
License for this repository is [BSD 3-Clause License](LICENSE).
We used [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) as our base code with BSD 3-Clause License [here](LICENSE.md).
