"""Render the tested 64×64 ResNet18 U-Net fixture as a PNG schematic."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).resolve().parent
im = Image.new('RGB', (1400, 920), '#f6f8fc')
d = ImageDraw.Draw(im)
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
def font(size):
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default(size=size)
def text(x, y, value, size=23, color='#172b4d', anchor='mm'):
    d.text((x, y), value, font=font(size), fill=color, anchor=anchor)
def arrow(points, color='#527493', width=4):
    d.line(points, fill=color, width=width)
    x,y=points[-1]; px,py=points[-2]; a=math.atan2(y-py,x-px)
    d.polygon([(x,y),(x-13*math.cos(a-.45),y-13*math.sin(a-.45)),
               (x-13*math.cos(a+.45),y-13*math.sin(a+.45))], fill=color)
def box(cx, cy, title, detail, fill='#e1edfb', stroke='#6087b6'):
    d.rounded_rectangle((cx-205,cy-36,cx+205,cy+36), 12, fill=fill, outline=stroke, width=2)
    text(cx,cy-12,title,24); text(cx,cy+17,detail,19)

text(700,40,'Native Earth Engine · ResNet18 U-Net',35)
text(700,81,'Two-channel 64 × 64 numerical fixture · 14,324,945 parameters',23,'#52637c')
left,right=300,1100
y=[225,335,445,555,665]
text(left,135,'Input: 2 × 64 × 64',23)
arrow([(left,156),(left,189)])
for a,b in zip(y,y[1:]): arrow([(left,a+36),(left,b-36)])
for a,b in zip(reversed(y),list(reversed(y))[1:]): arrow([(right,a-36),(right,b+36)],'#248579')
arrow([(left,701),(left,757),(right,757),(right,701)],'#248579')
text(700,783,'Nearest ×2 upsample → concatenate → two 3×3 convolutions',21,'#248579')
for i in range(4):
    arrow([(left+205,y[i]),(right-205,y[i+1])],'#96a6b9',3)
text(700,445,'Skip concatenations',20,'#62748b')
enc=[('Stem: 7×7 convolution, stride 2','64 × 32 × 32 · BN + ReLU'),
     ('MaxPool + residual stage 1','64 × 16 × 16 · 2 BasicBlocks'),
     ('Residual stage 2','128 × 8 × 8 · 2 BasicBlocks'),
     ('Residual stage 3','256 × 4 × 4 · 2 BasicBlocks'),
     ('Residual stage 4','512 × 2 × 2 · 2 BasicBlocks')]
dec=[('Decoder 0 · upsample, no skip','16 × 64 × 64'),
     ('Decoder 1 · upsample + stem skip','32 × 32 × 32'),
     ('Decoder 2 · upsample + stage 1','64 × 16 × 16'),
     ('Decoder 3 · upsample + stage 2','128 × 8 × 8'),
     ('Decoder 4 · upsample + stage 3','256 × 4 × 4')]
for cy,(title,detail) in zip(y,enc): box(left,cy,title,detail)
for cy,(title,detail) in zip(y,dec): box(right,cy,title,detail,'#dff2ee','#57a493')
arrow([(right,189),(right,156)],'#248579')
text(right,135,'1×1 head + sigmoid: 1 × 64 × 64',22)
text(700,832,'31 convolutions · 30 BatchNorms folded · residual additions + ReLU',23)
text(700,868,'Weights: FeatureCollection assets → native JSON decode → EE arrays',22,'#52637c')
im.save(OUT/'resnet18_unet_architecture.png')
print(OUT/'resnet18_unet_architecture.png')
