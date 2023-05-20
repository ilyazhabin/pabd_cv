import click
import tensorflow as tf
from model import make_model
from tensorflow import keras


data_dir ='data\\processed\\PetImages'
model_dir = 'models\\dummy_model'
image_size = (180, 180)
batch_size = 2
epochs = 2
lr = 1e-3

@click.command()
@click.option('-i', '--in_dir', default="data\\processed\\PetImages")
@click.option('-o', '--out_dir', default="models\\my_model")
@click.option('-e', '--epochs', default=4)
@click.option('-l', '--lr', default=1e-3)
@click.option('-b', '--batch_size', default=2)
@click.option('-s', '--image_size', default=180)
def train(in_dir, out_dir, epochs, lr, batch_size, image_size):
    image_size = (image_size, image_size)
    train_model(in_dir, out_dir, epochs, lr, batch_size, image_size)

def train_model(in_dir, out_dir, epochs, lr, batch_size, image_size):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        in_dir,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = make_model(input_shape=image_size + (3,), num_classes=2)


    callbacks = [
        # keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.save(out_dir)

if __name__ == '__main__':
    train()