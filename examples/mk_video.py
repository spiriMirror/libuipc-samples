import imageio as mio
import argparse as ap
import pathlib as pl

def main():
    parser = ap.ArgumentParser(description='Create video from images')
    parser.add_argument('folder', type=str, help='Folder contains images with numbered name. e.g. 0.png, 1.png, 2.png')
    parser.add_argument('fps', type=int, help='Frames per second')
    parser.add_argument('-o', '--output', type=str, help='Output video file', default='output.mp4')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    folder = args.folder
    output = args.output
    fps = args.fps
    verbose = args.verbose
    
    output = pl.Path(output).absolute()
    
    file_dict = {}
    images = []
    # find all files in the folder
    for f in pl.Path(folder).iterdir():
        if f.is_file():
            file_dict[int(f.stem)] = f

    # sort the files by number
    sorted_files = []
    for i in sorted(file_dict.keys()):
        sorted_files.append((i,file_dict[i]))
    
    # print the min and max number
    print(f'Number of images: {len(images)}')
    print(f'Min number: {sorted_files[0][0]}')
    print(f'Max number: {sorted_files[-1][0]}')
    
    # read the images
    for i, f in sorted_files:
        if(verbose):
            print(f'Reading {f}')
        images.append(mio.v2.imread(f))
    
    if(verbose):
        print(f'Image files:')
        for i,f in sorted_files:
            print(f'[{i}]: {f}')
    
    mio.mimsave(output, images, fps=fps)
    
    print(f'Video saved to {output}')
    
if __name__ == '__main__':
    main()