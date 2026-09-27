This is Multi Organ segmentation ini CT scans using SAM 2 segmentation usually used for video segmentation. SAM 2 was built to take into account the time axis of the video. Similarly we now leverage SAM2 to consider the depth of the given NII volume, and segment the organs
We have used nii volumes from nibabel, and have converted to image slices. We have then used bounding box annotation to finetune SAM2.hiera.small model's image encoder. ^ volumes are kept for validation
The flow is as fllows, we have divided the work into multiple weeks. Under the week 1 folder, The CT image preprocessor.py is present. Under the week2_updates folder, we have the functions used to create our dataset
in week3_updates folder we have the propogate and predict functions which are required to use the SAM 2 model
Seperately we have the lora_finetuning.py which has code for final finetuning of the SAM 2 using all of these functions. The data that is required for finetuning is present under the data folder

Results

Liver on 6 held-out BTCV scans (img0035–img0040), same box prompts for both models:

                                   Model	Dice ↑	           HD95 (mm) ↓
Zero-shot SAM 2.1 Hiera-S	          0.733 ± 0.144	          230.9 ± 45.6

SAM 2.1 Hiera-S + LoRA (this repo)	0.963 ± 0.003	           5.9 ± 1.8

Every scan improved; LoRA Dice ranges from 0.957 to 0.967.
Zero-shot's very high HD95 comes from stray fragments far from the organ. Fine-tuning removes them.
Only 119,808 parameters are trained, 0.26% of the model.

CT volume (.nii / .nii.gz)
  │  reorient to RAS, soft-tissue window −150…250 HU → uint8
  |
Training: individual axial slices --> LoRA fine-tuning of the image encoder
                                            │  (adapter weights only)
                                            |
Inference: all slices as video frames --> SAM 2 video predictor + LoRA
  │  one box prompt on one start slice per organ
  |
Bidirectional propagation along z (start → last slice, start → first slice)
  |
  |
3D organ mask  --> Dice, HD95

Preprocessing: Volumes are loaded with nibabel, reoriented so the third axis is axial, and clipped to a soft-tissue HU window.

Prompting: Each organ gets one box on one slice. The start slice is chosen from the 5 slices with the largest organ area: the one with the fewest disconnected pieces, then the most compact. The box is padded by 4 px.
LoRA fine-tuning:
Low-rank adapters (r = 8, α = 16) go into the attention qkv layers of the last 8 of the 16 Hiera blocks. Everything else in SAM 2 stays frozen.
Each slice is encoded once, and all of its organs (up to 6) are decoded from that one encoding.
Loss is BCE + per-mask soft Dice at 256 × 256, trained on all 13 BTCV organs.
3D inference:
The video predictor gets the box on the start slice and propagates forward and backward through the volume.
Each slice is conditioned on SAM 2's memory bank of already-segmented slices.
Slices are thresholded, resized to the original resolution and stacked into a 3D mask.
Evaluation: Volumetric Dice, and HD95 in millimetres using each scan's voxel spacing.

data:

ID Organ	     ID   Organ
1	spleen	      8	  aorta
2	right kidney	9  	inferior vena cava
3	left kidney	  10	portal and splenic vein
4	gallbladder	  11	pancreas
5	esophagus	    12	right adrenal gland
6	liver	        13	left adrenal gland
7	stomach

download the data from the data folder, first run it through CT_image_processor.py under week 1 folder, then use the week_2.py undedr week_2_updates. Then run the week_3.py using the outputs produced by previous functions
After this, we run the final finetuning.py which will finetune the sam model with the given data. Results and further explanatrion of our approach is given in the report which is also included in this repo
