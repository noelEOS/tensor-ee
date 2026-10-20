"""Render the approved diagram as a 17-second explanatory loop; not real timing.

Requires Pillow; the optional MP4 export also requires ffmpeg.
Run with Python from any directory.
The paired draw.py provides the editable source for the static figure.
"""
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from draw import Figure, BLUE, MUTED, S, draw_swin

BASE = draw_swin().im
FPS, SECONDS = 24, 17
DONE = '#357257'
DONE_FILL = '#eaf3ed'

def smooth(p):
    return (1-math.cos(math.pi*max(0,min(1,p))))/2

def travel(f, points, p):
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    remaining=sum(lengths)*smooth(p)
    visible=[points[0]]
    for a,b,length in zip(points,points[1:],lengths):
        u=min(1,remaining/length)
        visible.append((a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1])))
        remaining-=length
        if remaining<=0: break
    f.path(visible,BLUE,arrow=False,width=4)
    x,y=visible[-1]
    f.d.ellipse(((x-5)*S,(y-5)*S,(x+5)*S,(y+5)*S),fill=BLUE)

def active(f,index,p):
    x=[180,530,880][index]
    title=['Early encoder','Final encoder','Decoder + head'][index]
    detail=['Stages 1–3','Stage 4','UPerNet'][index]
    f.rect(x,340,230,95,BLUE,BLUE)
    f.text(x+115,383.5,title,24,'white',True,'center')
    f.text(x+115,413.5,detail,20,'#e4edf7',align='center')
    # A quiet time cue inside the active block; never interpreted as measured progress.
    f.path([(x+14,428),(x+14+202*p,428)],'#c8dcf2',arrow=False,width=2)

def check(f,x,y):
    f.path([(x,y-5),(x+5,y),(x+14,y-11)],DONE,arrow=False,width=2)

def completed(f,index):
    x=[180,530,880][index]
    title=['Early encoder','Final encoder','Decoder + head'][index]
    detail=['Stages 1–3','Stage 4','UPerNet'][index]
    f.rect(x,340,230,95,DONE_FILL,DONE)
    f.text(x+115,383.5,title,24,DONE,True,'center')
    f.text(x+115,413.5,detail,20,MUTED,align='center')
    f.text(x,317,['A','B','C'][index],20,DONE,True)
    check(f,x+28,315)
    f.text(x+50,317,'Done',18,DONE,True)

def saved(f,index):
    x=[180,530,880][index]
    title=['C1 · C2 · C3','C4','Prediction'][index]
    detail=['EE image · 56 × 56','EE image · 7 × 7','EE image · 224 × 224'][index]
    f.rect(x,500,223,88,DONE_FILL,DONE,3)
    f.text(x+111.5,534,title,23,DONE,True,'center')
    f.text(x+111.5,563,detail,19,MUTED,align='center')
    check(f,x+151,618)
    f.text(x+223,620,'Saved',18,DONE,True,'right')

def render(t):
    f=Figure('frame','Swin staged inference',820)
    f.im=BASE.copy(); f.d=ImageDraw.Draw(f.im)
    f.d.rectangle((0,708*S,1280*S,820*S),fill='white')
    status='One chip. Three execution steps. All inference runs in Earth Engine.'
    if .8<=t<2:
        status='A · Read the input chip and model weights.'
    elif 2<=t<3.6:
        status='A · Compute the early encoder features.'
    elif 3.6<=t<4.4:
        status='A · Export C1, C2 and C3 to an EE image asset.'
    elif 4.4<=t<5.2:
        status='A · Export complete. Saved features can now be read.'
    elif 5.2<=t<6.2:
        status='B · Read saved C3 and the next set of weights.'
    elif 6.2<=t<7.6:
        status='B · Compute the final encoder features.'
    elif 7.6<=t<8.3:
        status='B · Export C4 to an EE image asset.'
    elif 8.3<=t<9:
        status='B · Export complete. Both feature assets are ready.'
    elif 9<=t<10.6:
        status='C · Read C4 and the earlier features, plus decoder weights.'
    elif 10.6<=t<12.2:
        status='C · Combine the features and compute the prediction.'
    elif 12.2<=t<13:
        status='C · Export the prediction to an EE image asset.'
    elif 13<=t<16:
        status='Complete · The prediction is saved in Earth Engine.'
    f.text(48,744,status,23,DONE if 13<=t<16 else BLUE,True)
    f.text(48,784,'Illustrative sequence; not execution timing.',19,MUTED)
    f.text(1232,784,'Local Python orchestrates the tasks.',19,MUTED,align='right')

    for index,start,end in [(0,2,3.6),(1,6.2,7.6),(2,10.6,12.2)]:
        if start<=t<end: active(f,index,(t-start)/(end-start))
        elif t>=end: completed(f,index)
    for index,at in enumerate([4.4,8.3,13]):
        if at<=t: saved(f,index)
    transfers=[
        (.8,2,[(58,404),(180,404)]),
        (.8,2,[(295,242),(295,340)]),
        (3.6,4.4,[(295,435),(295,500)]),
        (5.2,6.2,[(410,543),(465,543),(465,387),(530,387)]),
        (5.2,6.2,[(645,242),(645,340)]),
        (7.6,8.3,[(645,435),(645,500)]),
        (9,10.6,[(760,543),(815,543),(815,387),(880,387)]),
        (9,10.6,[(295,595),(295,655),(1160,655),(1160,387),(1110,387)]),
        (9,10.6,[(995,242),(995,340)]),
        (12.2,13,[(995,435),(995,500)]),
    ]
    for start,end,points in transfers:
        if start<=t<end: travel(f,points,(t-start)/(end-start))
    if t>=15:
        f.im=Image.blend(f.im,INTRO,smooth((t-15)/1.2))
    return f.im

# The loop starts and ends on a fully readable static diagram.
INTRO=BASE.copy()
d=ImageDraw.Draw(INTRO)
d.rectangle((0,708*S,1280*S,820*S),fill='white')
intro=Figure('intro','Swin',820); intro.im=INTRO; intro.d=d
intro.text(48,744,'One chip. Three execution steps. All inference runs in Earth Engine.',23,BLUE,True)
intro.text(48,784,'Illustrative sequence; not execution timing.',19,MUTED)
intro.text(1232,784,'Local Python orchestrates the tasks.',19,MUTED,align='right')

def gif_frame(t):
    # Hold the completed diagram, then reset. Avoid a costly whole-image GIF fade.
    return INTRO.copy() if t >= 16.2 else render(min(t, 14.9))


def write_gif(path):
    size, fps = (1600, 1025), 15
    # Quantize clean source frames, not compressed video. A shared palette keeps
    # stationary text identical between frames so GIF delta compression works.
    samples = [render(t).resize(size, Image.Resampling.LANCZOS)
               for t in [0, 2.7, 6.8, 10, 11.5, 13.75]]
    atlas = Image.new('RGB', (size[0], size[1] * len(samples)))
    for index, sample in enumerate(samples):
        atlas.paste(sample, (0, index * size[1]))
    palette = atlas.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
    frames = [gif_frame(index / fps).resize(size, Image.Resampling.LANCZOS)
              .quantize(palette=palette, dither=Image.Dither.NONE)
              for index in range(fps * SECONDS)]
    # GIF delays are centiseconds; alternate 60/70 ms to preserve 17 seconds.
    durations = [round((index + 1) * 100 / fps) * 10
                 - round(index * 100 / fps) * 10
                 for index in range(len(frames))]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    print(f'GIF rendered: {path.stat().st_size:,} bytes', flush=True)


def write_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '2560x1640', '-r', str(FPS), '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    for frame in range(FPS * SECONDS):
        encoder.stdin.write(render(frame / FPS).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('MP4 encoder failed')
    print('MP4 rendered.', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'docs' / 'images')
    parser.add_argument('--video', type=Path, help='Also export a full-resolution MP4 here')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_gif(args.output / 'swin-inference.gif')
    if args.video:
        write_video(args.video)


if __name__ == '__main__':
    main()
