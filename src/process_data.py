import os.path
import shutil
import tensorflow as tf

import click


# in_dir = 'data\\raw\\kaggle'
# out_dir = 'data\\processed\\PetImages'
# n_img = 20


@click.command()
@click.option('-i', '--in_dir', default='data\\raw\\kaggle')
@click.option('-o', '--out_dir', default='data\\processed\\PetImages')
@click.option('-n', '--n_img', default=20)
@click.option('-f','--filter_corrupted', default=False)
def process_data(in_dir, out_dir, n_img):
    # make_out_dir(out_dir)
    copy_imgs(in_dir, out_dir, n_img)


def make_out_dir(out_dir):
    if os.path.exists(out_dir):
        os.remove(out_dir)
        os.mkdir(out_dir)
        os.mkdir(os.path.join(out_dir, 'Cat'))
        os.mkdir(os.path.join(out_dir, 'Dog'))

def filter_corrupted(out_dir):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(out_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

def copy_imgs(in_dir, out_dir, n_img):
    all_imgs = os.listdir(in_dir)
    cat_imgs = [img for img in all_imgs if img.startswith('cat')]
    dog_imgs = [img for img in all_imgs if img.startswith('dog')]


    for cat_img in cat_imgs[:n_img]:
        shutil.copy2(
            os.path.join(in_dir, cat_img),
            os.path.join(out_dir, "Cat")
        )

    for dog_img in dog_imgs[:n_img]:
        shutil.copy2(
            os.path.join(in_dir, dog_img),
            os.path.join(out_dir, "Dog")
        )


if __name__ == '__main__':
    process_data()