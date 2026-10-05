"""Compose the official marks without changing their paths or proportions.

Requires matplotlib, fontTools, CairoSVG, and Noto Sans CJK SC for Chinese text.
Pass the font file as the only argument. The site uses the same original marks.
"""
from pathlib import Path
import copy,io,sys,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import cairosvg
ROOT=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
def el(tag,attrs,parent=None):
 e=ET.Element('{'+NS+'}'+tag,{k:str(v) for k,v in attrs.items()})
 if parent is not None:parent.append(e)
 return e
font=FontProperties(fname=sys.argv[1])
english=FontProperties(family='DejaVu Sans',weight='bold')
s=el('svg',{'width':1300,'height':320,'viewBox':'0 0 1300 320','role':'img','aria-labelledby':'title desc'})
t=el('title',{'id':'title'},s);t.text='清华大学 × 华为 · Tsinghua University × Huawei Technologies'
t=el('desc',{'id':'desc'},s);t.text='Official vector institution marks. Paper authors are affiliated with Tsinghua University and Huawei Technologies.'
el('rect',{'width':1300,'height':320,'fill':'white'},s)
def logo(name,x,y,size):
 orig=ET.parse(ROOT/name).getroot();orig.attrib.update({'x':str(x),'y':str(y),'width':str(size),'height':str(size)})
 for parent in orig.iter():
  for child in list(parent):
   if child.tag.endswith('style'):parent.remove(child)
 for element in orig.iter():
  if 'id' in element.attrib:element.set('id',name.split('.')[0]+'-'+element.get('id'))
  for key,value in list(element.attrib.items()):
   if 'url(#' in value:element.set(key,value.replace('url(#','url(#'+name.split('.')[0]+'-'))
 s.append(orig)
def label(value,x,y,size,color,prop):
 path=TextPath((0,0),value,size=size,prop=prop)
 from matplotlib.backends.backend_svg import _path
 data=_path.convert_to_string(path,None,None,False,None,6,[b'M',b'L',b'Q',b'C',b'z'],False).decode()
 el('path',{'d':data,'transform':f'translate({x} {y}) scale(1 -1)','fill':color},s)
logo('tsinghua-emblem.svg',66,55,175)
label('清华大学',272,136,62,'#702082',font)
label('TSINGHUA',275,181,25,'#702082',english)
label('UNIVERSITY',275,215,25,'#702082',english)
label('×',625,163,50,'#b2a9ba',FontProperties(family='DejaVu Sans'))
logo('huawei-logo.svg',735,55,175)
label('华为',961,136,62,'#171320',font)
label('HUAWEI',963,181,25,'#171320',english)
label('TECHNOLOGIES',963,215,25,'#171320',english)
el('rect',{'x':66,'y':280,'width':569,'height':3,'fill':'#702082'},s)
el('rect',{'x':665,'y':280,'width':569,'height':3,'fill':'#cf1734'},s)
data=ET.tostring(s,encoding='utf-8',xml_declaration=True)
(ROOT/'brand-banner.svg').write_bytes(data)
cairosvg.svg2png(bytestring=data,write_to=str(ROOT/'brand-banner-3x.png'),scale=3)
