""" USAGE
python main.py train --image ./images/train/stone.png --num_scales 8

python main.py inference --image ./images/train/stone_editing.png --dir ./training_checkpoints  --mode editing  --inject_scale 2
 """
import os
import argparse
import tensorflow as tf

from train import Trainer
from inference import Inferencer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SinGAN')
    parser.add_argument('command',help="'train' or 'inference'")
    parser.add_argument('--image', help='Input image name', required=True)
    parser.add_argument('--dir', help='Checkpoints directory', default='./training_checkpoints')
    parser.add_argument('--image_size', type=int, help='New size of training/inference image', default=250)
    parser.add_argument('--scale_factor', type=float, help='Pyramid scale factor', default=0.75)

    # Training arguments
    parser.add_argument('--num_scales', type=int, default=8)
    parser.add_argument('--num_iters', type=int, help='Number of iteration per scale', default=2000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_size', type=int, default=250)
    parser.add_argument('--min_size', type=int, default=25)
    parser.add_argument('--debug', type=bool, help='Whether to pring the loss', default=True)

    # Inference arguments
    parser.add_argument('--num_samples', type=int, help='Number of random samples to generate', default=50)
    parser.add_argument('--inject_scale', type=int, help='The scale to start generating', default=0)
    parser.add_argument('--result_dir', help='Results directory', default='./results')
    parser.add_argument('--mode', default='random_sample',
                        help='Inference mode: random_sample, harmonization, paint2image, editing')

    args = parser.parse_args()


    # Validate arguments
    if args.command == 'train':
        assert os.path.exists(args.image), 'Training image not found !'
        assert args.image_size >= 0
        assert args.num_scales > 0
        assert args.num_iters > 0
        assert args.scale_factor > 0
        assert args.learning_rate > 0
        assert args.max_size > 0
        assert args.min_size > 0

        parameters = {
                'checkpoint_dir' : args.dir,
                'num_scales' : args.num_scales,
                'num_iters' : args.num_iters,
                'max_size' : args.max_size,
                'min_size' : args.min_size,
                'scale_factor' : args.scale_factor,
                'learning_rate' : args.learning_rate,
                'debug' : args.debug
            }

        trainer = Trainer(**parameters)
        trainer.train(training_image=args.image)


    elif args.command == 'inference':
        assert os.path.exists(args.image), 'Reference image not found !'
        assert os.path.exists(args.dir), "Model doesn't exist, please train first"
        assert (args.mode == 'random_sample') or (args.mode == 'harmonization') or (args.mode == 'paint2image') or (args.mode == 'editing'), 'Inference mode: random_sample, harmonization, paint2image, editing'
        assert args.inject_scale >= 0
        assert args.image_size >= 0

        parameters = {
                'num_samples' : args.num_samples,
                'scale_factor' : args.scale_factor,
                'inject_scale' : args.inject_scale,
                'result_dir' : args.result_dir,
                'checkpoint_dir' : args.dir,
            }

        inferencer = Inferencer(**parameters)
        inferencer.inference(mode=args.mode, reference_image=args.image, image_size=args.image_size)


    else:
        print('Example usage : python main.py train --image ./images/target/rock.jpg')
        

if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    # )
    
    main()