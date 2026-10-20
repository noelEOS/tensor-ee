"""Editable sources for the README figures; no Earth Engine calls.

Uses Pillow, as the other diagram renderers do. Coordinates are in a 1280-pixel
wide layout; raster figures are drawn at twice that resolution.
"""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

S = 2
INK = '#24292f'
MUTED = '#59636e'
BLUE = '#3265a1'
TINT = '#edf3fa'
LINE = '#b9c2cc'
LIGHT = '#f6f7f8'

class Figure:
    def __init__(self, name, title, h=800):
        self.name, self.h = name, h
        self.im = Image.new('RGB', (1280*S,h*S), 'white')
        self.d = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{h}" viewBox="0 0 1280 {h}" role="img"><title>{escape(title)}</title><rect width="1280" height="{h}" fill="white"/>']
    def rect(self,x,y,w,h,fill='white',stroke=LINE,r=5):
        self.d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),radius=r*S,fill=fill,outline=stroke,width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
    def text(self,x,y,t,size=22,color=INK,bold=False,align='left'):
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size*S)
        anchor={'left':'ls','center':'ms','right':'rs'}[align]
        self.d.text((x*S,y*S),t,font=font,fill=color,anchor=anchor)
        weight='700' if bold else '400'
        sa={'left':'start','center':'middle','right':'end'}[align]
        self.svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{sa}">{escape(t)}</text>')
    def path(self,pts,color=MUTED,dash=False,arrow=True,width=2):
        for (x,y),(xx,yy) in zip(pts,pts[1:]):
            dist=math.hypot(xx-x,yy-y)
            if dash:
                for start in range(0,int(dist),12):
                    a=start/dist;b=min(start+6,dist)/dist
                    self.d.line(((x+(xx-x)*a)*S,(y+(yy-y)*a)*S,(x+(xx-x)*b)*S,(y+(yy-y)*b)*S),fill=color,width=width*S)
            else:self.d.line((x*S,y*S,xx*S,yy*S),fill=color,width=width*S)
        coords=' '.join(f'{x},{y}' for x,y in pts)
        ds=' stroke-dasharray="6 6"' if dash else ''
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{ds}/>')
        if arrow:
            (x,y),(px,py)=pts[-1],pts[-2]
            a=math.atan2(y-py,x-px)
            tri=[(x,y),(x-10*math.cos(a-.45),y-10*math.sin(a-.45)),(x-10*math.cos(a+.45),y-10*math.sin(a+.45))]
            self.d.polygon([(a*S,b*S) for a,b in tri],fill=color)
            self.svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in tri)+f'" fill="{color}"/>')
    def box(self,x,y,w,h,title,detail='',blue=False):
        self.rect(x,y,w,h,TINT if blue else 'white',BLUE if blue else LINE)
        self.text(x+w/2,y+(h/2-4 if detail else h/2+8),title,24,bold=True,align='center')
        if detail:self.text(x+w/2,y+h/2+26,detail,20,MUTED,align='center')
    def asset(self,x,y,w,title,detail):
        self.rect(x+7,y+7,w-7,88,LIGHT,LINE,3)
        self.rect(x,y,w-7,88,'white',LINE,3)
        self.text(x+(w-7)/2,y+34,title,23,bold=True,align='center')
        self.text(x+(w-7)/2,y+63,detail,19,MUTED,align='center')
    def save(self, directory):
        self.svg.append('</svg>')
        (directory/(self.name+'.svg')).write_text('\n'.join(self.svg))
        self.im.save(directory/(self.name+'.png'))



def draw_comparison():
    f=Figure('inference-location','Where inference runs',800)
    f.text(48,60,'Where inference runs',36,bold=True)
    f.text(48,96,'The same starting point: imagery already in Earth Engine.',23,MUTED)
    for x,title in [(48,'Earth Engine + Vertex AI'),(664,'Earth Engine + TensorEE')]:
        f.text(x,154,title,27,bold=True)

    # A repeated input/output shape makes the execution boundary the comparison.
    f.rect(48,178,568,105,LIGHT,'#e1e5e9')
    f.rect(48,581,568,105,LIGHT,'#e1e5e9')
    f.rect(664,178,568,508,LIGHT,'#e1e5e9')
    f.box(76,194,512,72,'Input imagery','Earth Engine')
    f.box(692,194,512,72,'Input imagery','Earth Engine')
    f.path([(332,266),(332,368)])
    f.text(355,310,'Image tiles',20,MUTED)
    f.text(355,340,'fromVertexAi · predictImage',18,MUTED)
    f.rect(48,368,568,132,'white',LINE)
    f.text(70,393,'VERTEX AI',16,MUTED,bold=True)
    f.box(76,407,512,72,'Deployed model endpoint','CPU / optional GPU')
    f.path([(332,500),(332,597)])
    f.text(355,554,'Predictions',20,MUTED)
    f.box(76,597,512,72,'Prediction image','Earth Engine')

    f.path([(948,266),(948,407)],BLUE)
    f.text(970,335,'Native EE graph',20,BLUE)
    f.box(692,407,512,72,'Model forward pass','EE image and array operations',blue=True)
    f.path([(948,479),(948,597)],BLUE)
    f.box(692,597,512,72,'Prediction image','Earth Engine')
    f.text(332,735,'A separate model-serving deployment',22,MUTED,align='center')
    f.text(948,735,'Inference stays inside Earth Engine',22,BLUE,bold=True,align='center')
    f.text(48,782,'TensorEE supports selected models; larger graphs may execute in stages.',19,MUTED)
    return f


def draw_swin():
    f=Figure('swin-staged-inference','Swin inference, in three steps',820)
    f.text(48,60,'Swin inference, in three steps',36,bold=True)
    f.text(48,96,'Fixed Swin-T / UPerNet · one 224 × 224 chip',23,MUTED)
    f.rect(36,130,1208,570,LIGHT,'#e1e5e9')
    f.text(59,162,'EARTH ENGINE',16,MUTED,bold=True)
    f.rect(180,187,930,55,'white',LINE,3)
    f.text(645,222,'Model weights  ·  EE table assets',23,bold=True,align='center')
    for x in [295,645,995]:
        f.path([(x,242),(x,340)],MUTED,True)
    f.text(180,317,'A',20,BLUE,True)
    f.text(530,317,'B',20,BLUE,True)
    f.text(880,317,'C',20,BLUE,True)
    f.box(180,340,230,95,'Early encoder','Stages 1–3',True)
    f.box(530,340,230,95,'Final encoder','Stage 4',True)
    f.box(880,340,230,95,'Decoder + head','UPerNet',True)
    f.text(110,357,'224 × 224',19,MUTED,align='center')
    f.text(110,382,'input',19,MUTED,align='center')
    f.path([(58,404),(180,404)],BLUE)
    for x in [295,645,995]:
        f.path([(x,435),(x,500)],BLUE)
        f.text(x+15,474,'Export',18,MUTED)
    f.asset(180,500,230,'C1 · C2 · C3','EE image · 56 × 56')
    f.asset(530,500,230,'C4','EE image · 7 × 7')
    f.asset(880,500,230,'Prediction','EE image · 224 × 224')
    f.path([(410,543),(465,543),(465,387),(530,387)],BLUE)
    f.text(476,465,'C3',17,BLUE)
    f.path([(760,543),(815,543),(815,387),(880,387)],BLUE)
    f.path([(295,595),(295,655),(1160,655),(1160,387),(1110,387)],BLUE)
    f.text(680,644,'Earlier features also feed the decoder',20,BLUE,align='center')
    f.text(48,741,'Local Python builds requests and tracks tasks.',22,MUTED)
    f.text(48,779,'Weights and intermediate features stay in EE assets; no GCS bucket is needed.',21,MUTED)
    return f


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'docs' / 'images')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    draw_comparison().save(args.output)
    draw_swin().save(args.output)


if __name__ == '__main__':
    main()
