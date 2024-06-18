from model import create_model
model1=create_model()


model1.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))
model1.fit(train_low_images, train_high_images,
                        batch_size=16,
                        validation_data=val_dataset,
                        # steps_per_epoch=steps_per_epoch_train,
                        # validation_steps=steps_per_epoch_validation,
                        epochs=150,
                        verbose=1,
                        # callbacks=callbacks_lst
          )
model_save_path = '/content/drive/My Drive/model_weights3.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save model weights
model1.save_weights(model_save_path)
print(f'Model weights saved to {model_save_path}')


