# T5 Textual Inversion for [DeepFloyd IF](https://github.com/deep-floyd/IF) on a 24 GB GPU
[*T5_Inversion.ipynb*](./T5_Inversion.ipynb) is **Copyright © 2023 [HANS ROETTGER](mailto:oss.roettger@posteo.org)** and distributed under the terms of **[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html)**.  

This is an implementation of the textual inversion algorithm to incorporate your own objects, faces, logos or styles into DeepFloyd IF.  
Input: a a couple of original images. Output: an embedding for a single token, that can be used in the standard DeepFloyd IF dream pipeline to generate your artefacts.

**I know you don't care about copyright, but at least leave me a ⭐ star in the top right!**

## Original Images ➡ T5 Embedding

<img src="./samples/input.png" alt="" border=3></img>

Run the notebook to generate and save the T5 Embedding for your images. It takes about 35 minutes on a RTX 3090 GPU.

## T5 Embedding ➡ Deep Floyd IF Dream Pipeline
Load the T5 Embedding to a single token (e.g. "my") and use it in the standard DeepFloyd IF dream prompts.  

    load_embedding(t5,word="my",embedding_file="myPuppet.pt",path="./Embeddings/")
    prompts=["a photo of {} woman at the beach".format("my")]

<table style="width: 100%">
<tr>
    <td colspan=2><img src="./samples/2a.png" alt="" height=128 width=128 border=3></img></td>
    <td colspan=2><img src="./samples/2b.png" alt="" height=128 width=128 border=3></img></td>
    <td colspan=2><img src="./samples/2c.png" alt="" height=128 width=128 border=3></img></td>
    <td colspan=2><img src="./samples/2d.png" alt="" height=128 width=128 border=3></img></td>
    </tr>
</table>

## Prerequisites
* A working  [DeepFloyd IF](https://github.com/deep-floyd/IF) environment
* A GPU with at least 24 GB CUDA memory

## Installation
* Copy the [*T5_Inversion.ipynb*](./T5_Inversion.ipynb) notebook into your DeepFloyd IF environment. All you need is in the small notebook.
* Set the paths to your local models at the top of the notebook. Restart and run all!
* The t5-v1_1-xxl/IF-I-M-v1.0 models in training mode closely fit into 24 GB CUDA memory.
* If you run into out of memory errors, try to remove the "safety_checker" & "watermarker" from the "model_index.json" files to save up some memory (see [template](./model_index.json) in this repository).

## Issues
* Since DeepFloyd IF uses a puny 64x64 pixel resolution in Stage_I, the results are not as good as textual inversion for stable diffusion, but far better than expected.
* I haven't explored all parameter combinations yet - especially the learning rates are tricky. Take this implementation as a starting point for your own experiments.
* T5 embeddings trained with the small IF-I-M-v1.0 model generate good images with the same Stage_I model only! You can try those with the XL model, but you will see other results.
* If you get totally blurred results, DeepFloyd IF has detected "nudity". You can remove line 226 in /deepfloyd_if/modules/base.py #sample = self.__validate_generations(sample) to prevent that.
* I couldn't try out the T5 Inversion with the IF-I-XL-v1.0 model due to CUDA memory restrictions. Maybe there is someone out there with a RTX6000, A100 or H100? Feedback welcome!


