mkdir -p /data02/Text2Video/annotations
cp -r /root/200w /data02/Text2Video/annotations/
cp -r /root/200w_nobody /data02/Text2Video/annotations/
cp -r /root/150w /data02/Text2Video/annotations/
cp -r /root/koala /data02/Text2Video/annotations/
mkdir -p /data02/model_zoo/huggingface/hunyuan
cp -r /root/hunyuanvideo_13b /data02/model_zoo/huggingface/hunyuan/
# cp -r /root/text_encoder /data02/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder
# cp -r /root/text_encoder_2 /data02/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/text_encoder_2