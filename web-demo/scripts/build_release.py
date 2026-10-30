"""Bind HTML, metadata, predictions and content-addressed videos into one release."""
from pathlib import Path
import argparse,base64,hashlib,json,re
from bs4 import BeautifulSoup
parser=argparse.ArgumentParser();parser.add_argument('--static-dir',default=str(Path(__file__).resolve().parents[1]/'dist'));args=parser.parse_args()
root=Path(args.static_dir).resolve();assets=root/'assets';assets.mkdir(exist_ok=True)
hashof=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((root/'samples.json').read_text());samples=manifest['samples']
entries=[]
for sample in samples:
 item=dict(sample);media=root/sample['file'];raw=media.read_bytes();sha=hashof(raw)
 destination=root/'media'/f"{sample['id']}.{sha[:12]}.mp4"
 if media!=destination:
  destination.write_bytes(raw)
 sample['file']=item['file']=str(destination.relative_to(root))
 item['bytes']=len(raw);item['sha256']=sha
 item['sources']=[item['file'], *manifest.get('media_mirrors',{}).get(sha,[])]
 item['poster']='data:image/jpeg;base64,'+base64.b64encode((root/sample['poster']).read_bytes()).decode()
 item['record']=json.loads((root/sample['recording']).read_text())
 entries.append(item)
(root/'samples.json').write_text(json.dumps(manifest,indent=2)+'\n')
seed=json.dumps(entries,sort_keys=True,separators=(',',':')).encode();release='collection-'+hashof(seed)[:12]
data={'schema':2,'release':release,'samples':entries};payload=json.dumps(data,separators=(',',':')).replace('<','\\u003c')
collection_file=f'collection.{hashof(payload.encode())[:12]}.json';(assets/collection_file).write_text(payload)
js=(Path(__file__).parent/'demo-runtime.js').read_text().replace('__RELEASE__',release).replace('__COLLECTION_URL__','assets/'+collection_file).replace('__SAMPLE_COUNT__',str(len(entries)))
(root/'demo.js').write_text(js);jsfile=f'app.{hashof(js.encode())[:12]}.js';(assets/jsfile).write_text(js)
css=(root/'style.css').read_bytes();cssfile=f'style.{hashof(css)[:12]}.css';(assets/cssfile).write_bytes(css)
soup=BeautifulSoup((root/'index.html').read_text(),'html.parser')
for elem in soup.find_all('script'):elem.decompose()
for elem in soup.select('link[rel="stylesheet"]'):elem['href']='assets/'+cssfile
old=soup.find(id='preload-bin')
if old:old.decompose()
script=soup.new_tag('script',id='example-data',type='application/json');script.string=payload;soup.body.append(script)
script=soup.new_tag('script',src='assets/'+jsfile,defer='');soup.body.append(script)
soup.html['data-release']=release
soup.find(id='player-status').string='Choose any example to play.'
soup.find(id='cache-label').string=f'{len(entries)} examples · choose any to play'
soup.find(id='video-name').string='Choose an example'
soup.find(id='prepare-text').string='Choose an example to begin'
(root/'index.html').write_text(str(soup))
# Retain immutable assets: previously cached entry documents must keep working
# after future releases. Old aliases also remain available for compatibility.
print(json.dumps({'release':release,'app':'assets/'+jsfile,'styles':'assets/'+cssfile,'collection':'assets/'+collection_file,'examples':len(entries),'html_bytes':(root/'index.html').stat().st_size}))
