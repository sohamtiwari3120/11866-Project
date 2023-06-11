python predict_autoregressive_predictor.py \
--config /home/ubuntu/learning2listen/src/configs/vq/delta_v6.json \
--checkpoint /home/ubuntu/learning2listen/src/models/delta_v6_wo_text_es200_er2er_best.pth \
--etag gen_inf_selected_video_wo_txt \
--save \
--save_folder ./inference_outputs/neg_2_copy_wo_text/ \
-ivp /home/ubuntu/learning2listen/src/data/reference_style_embedidngs/result_cropped/neg_1_cropped_video_embeddings.npy \
-iap /home/ubuntu/learning2listen/src/data/reference_style_embedidngs/result_cropped/neg_1_cropped_audio.npy \
-itp /home/ubuntu/learning2listen/src/data/reference_style_embedidngs/result_cropped/neg_1_KateSchatzOnTalkingToPeopleWhoDontAgreeWithYouCONANonTBSvO1rqi19Tjk.txt.npy \
