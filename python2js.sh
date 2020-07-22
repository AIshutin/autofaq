tensorflowjs_converter \
    --input_format=tf_hub \
    --saved_model_tags="train" \
    --signature_name="tokens" \
    https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1 \
    mobilebert/web_model
