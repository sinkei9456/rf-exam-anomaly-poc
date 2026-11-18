"""doc/00_overview.md 를 읽기용 html(doc/00_overview.html)로 바꾼다. md 는 건드리지 않는다. (2026-11-18, research/shoes/src/build_html.py 와 같은 방식)

* 그림은 html 안에 넣는다(base64). 그래서 html 파일 하나만 옮겨도 그림이 보인다.
* 그림 바로 뒤의 "그림 ..." 또는 "**그림 ...**" 문단은 그림 설명으로 묶는다.
* 문서 사이 링크는 빼고 글자만 남긴다. 같은 문서 안으로 가는 링크(#...)는 남긴다.
* 목록 앞에 빈 줄이 없으면 넣고, 공백 2-3칸으로 들여쓴 하위 목록은 4칸으로 바꾼다(html 쪽만).
* 왼쪽에 목차(## 절과 ### 소절)를 둔다. 누르면 그 절로 가고, 화면을 내려도 목차는 그 자리에 있으며
  지금 읽는 절을 진하게 표시한다. 폭이 좁은 화면(1,100 px 미만)에서는 본문 위에 접을 수 있는 목차로 나온다.
* 절 주소(id)는 영문·숫자만 쓴다: "4.5 (마) ..." → s4-5, 번호 없는 제목 → h. 코드 서버(VS Code) 미리보기는 링크의 href 글자가
  브라우저가 계산한 # 뒷부분과 똑같을 때만 문서 안에서 이동하고, 다르면 바깥 새 탭으로 연다. 한글 주소는 %EB%A7%88 처럼 바뀌어
  달라지므로 쓰면 안 된다(2026-11-18 고침). 클릭 처리 JS 도 코드 서버 처리기까지 클릭이 가지 않게 막는다(stopPropagation).

실행: python src/build_html.py   (requirements.txt 의 markdown 모듈이 필요)
00_overview.md 를 고친 뒤에는 다시 돌린다.
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "doc" / "00_overview.md"
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
CSS = """
:root{--bg:#fcfcfb;--ink:#1b1b1a;--ink2:#5c5b57;--line:#c9c8c2;--head:#f1f0ec;--rule:#ddd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#ecebe6;--ink2:#a9a8a2;--line:#555;--head:#2a2a28;--rule:#444}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#ecebe6;--ink2:#a9a8a2;--line:#555;--head:#2a2a28;--rule:#444}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Noto Sans CJK KR','Noto Sans KR','NanumGothic','Malgun Gothic',sans-serif;font-size:16px;line-height:1.7}
:lang(ko){word-break:keep-all}

.layout{display:grid;grid-template-columns:minmax(0,960px);justify-content:center;gap:40px;padding:0 16px}
main{min-width:0;padding:32px 0 80px;overflow-wrap:break-word}
h2,h3{scroll-margin-top:16px}
nav.toc{font-size:.85em;line-height:1.45}
nav.toc details{border:1px solid var(--rule);border-radius:6px;padding:8px 12px;margin-top:16px;background:var(--head)}
nav.toc summary{cursor:pointer;font-weight:700}
nav.toc ul{list-style:none;margin:6px 0 0;padding:0}
nav.toc ul ul{margin:2px 0 4px;padding-left:14px}
nav.toc li{margin:2px 0}
nav.toc a{display:block;padding:2px 6px;border-radius:4px;color:var(--ink2);text-decoration:none}
nav.toc a:hover{background:var(--rule);color:var(--ink)}
nav.toc a.active{color:var(--ink);font-weight:700;background:var(--rule)}
@media (min-width:1100px){
.layout{grid-template-columns:260px minmax(0,960px)}
nav.toc{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;padding:32px 0 32px}
nav.toc details{margin-top:0;background:none;border:none;padding:0}
nav.toc summary{list-style:none;pointer-events:none;margin-bottom:4px}
nav.toc summary::-webkit-details-marker{display:none}
}
@media print{nav.toc{display:none}.layout{display:block;padding:0}}
h1{font-size:1.6em;line-height:1.3}
h2{font-size:1.25em;margin-top:2em;border-bottom:1px solid var(--rule);padding-bottom:4px}
h3{font-size:1.05em;margin-top:1.5em}
figure{margin:24px 0}
img{display:block;max-width:100%;height:auto;margin:0 auto;background:#fff;border:1px solid var(--rule)}
figcaption{margin-top:8px;font-size:.9em;line-height:1.55;color:var(--ink2)}
table{display:block;overflow-x:auto;border-collapse:collapse;font-size:.85em;line-height:1.45;margin:12px 0}
th,td{border:1px solid var(--line);padding:4px 8px;vertical-align:top}
th{background:var(--head)}
code{font-size:.9em;background:var(--head);padding:1px 4px;border-radius:3px}
pre{background:var(--head);padding:12px;overflow-x:auto;line-height:1.4}
pre code{background:none;padding:0}
"""
SCRIPT = """
(function(){
  var links=[].slice.call(document.querySelectorAll('nav.toc a'));
  var heads=links.map(function(a){return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));});
  function mark(){
    var cur=0;
    for(var i=0;i<heads.length;i++){ if(heads[i] && heads[i].getBoundingClientRect().top<=80){cur=i;} }
    if(window.innerHeight+window.scrollY>=document.documentElement.scrollHeight-2){
      var h=decodeURIComponent(location.hash.slice(1)), k=-1;
      for(var j=0;j<heads.length;j++){ if(heads[j] && heads[j].id===h){k=j;} }
      if(k>=0 && heads[k].getBoundingClientRect().top<window.innerHeight){cur=k;}
      else{ for(var j2=0;j2<heads.length;j2++){ if(heads[j2] && heads[j2].getBoundingClientRect().top<window.innerHeight){cur=j2;} } }
    }
    links.forEach(function(a,i){a.classList.toggle('active',i===cur);});
    var act=links[cur], nav=document.querySelector('nav.toc');
    if(act && nav && nav.scrollHeight>nav.clientHeight){
      var r=act.getBoundingClientRect(), n=nav.getBoundingClientRect();
      if(r.top<n.top+40||r.bottom>n.bottom-40){nav.scrollTop+=r.top-n.top-n.height/2;}
    }
  }
  window.addEventListener('scroll',mark,{passive:true}); window.addEventListener('resize',mark);
  window.addEventListener('load',function(){mark(); setTimeout(mark,150);}); window.addEventListener('hashchange',function(){setTimeout(mark,50);});
  // 목차를 누르면 페이지를 이동하지 않고 그 절로 화면만 내린다. 코드 서버 미리보기는 창 단위 클릭 처리기가 링크를 바깥 새 탭으로
  // 열 수 있어서, 클릭이 그 처리기까지 가지 않게 막는다(stopPropagation).
  links.forEach(function(a,i){
    a.addEventListener('click',function(e){
      var h=heads[i]; if(!h){return;}
      e.preventDefault(); e.stopPropagation();
      h.scrollIntoView({behavior:'instant',block:'start'});
      try{ history.replaceState(null,'','#'+encodeURIComponent(h.id)); }catch(err){}
      mark();
    });
  });
  mark();
})();
"""
LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def ascii_id(value, separator):
    """제목 → 영문·숫자 주소. "4.5 (마) CNN·CAE 앙상블" → s4-5, "0. 한눈에 보기" → s0, 번호 없는 제목 → h(겹치면 toc 가 _1 을 붙인다)."""
    m = re.match(r"\s*(\d+(?:\.\d+)*)", value)
    return "s" + m.group(1).replace(".", "-") if m else "h"


def md_fix(text):
    """목록 앞에 빈 줄을 넣고, 공백 2-3칸으로 들여쓴 하위 목록을 4칸으로 바꾼다. 코드 블록 안은 그대로 둔다."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code:
            line = re.sub(r"^ {2,3}(?=([-*]|\d+\.)\s)", "    ", line)
            if LIST.match(line) and out and out[-1].strip() and not LIST.match(out[-1]) and not out[-1].startswith("    "):
                out.append("")
        out.append(line)
    return "\n".join(out)


def build(md_path):
    base = md_path.parent

    def embed(m):
        alt, src, caption = m.group(1), m.group(2), m.group(3)
        image = (base / src).resolve()
        data = base64.b64encode(image.read_bytes()).decode()
        return (f'<figure><img alt="{alt}" src="data:{MIME[image.suffix.lower()]};base64,{data}">'
                f"<figcaption>{caption}</figcaption></figure>")

    md = markdown.Markdown(extensions=["tables", "sane_lists", "fenced_code", "toc"],
                           extension_configs={"toc": {"slugify": ascii_id, "toc_depth": "2-3"}})
    html = md.convert(md_fix(md_path.read_text(encoding="utf-8")))
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>\s*<p>((?:<strong>)?그림 .*?)</p>', embed, html, flags=re.S)
    if re.search(r'<img [^>]*src="(?!data:)', html):
        raise SystemExit(f"{md_path.name}: 바로 뒤에 '그림 ...' 설명이 없는 그림이 있음")
    html = re.sub(r'<a href="(?![a-z]+:|#)[^"]*">(.*?)</a>', r"\1", html, flags=re.S)
    title = re.sub(r"<[^>]+>", "", re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S).group(1))
    toc = re.sub(r'^<div class="toc">\s*|\s*</div>\s*$', "", md.toc.strip())
    nav = f'<nav class="toc" aria-label="목차"><details open><summary>목차</summary>\n{toc}\n</details></nav>'
    page = (f'<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{title}</title>\n'
            f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"layout\">\n{nav}\n<main>\n{html}\n</main>\n</div>\n"
            f"<script>{SCRIPT}</script>\n</body>\n</html>\n")
    target = md_path.with_suffix(".html")
    target.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    t = build(SOURCE)
    print(f"{t.relative_to(ROOT)}  {t.stat().st_size / 1e6:.1f} MB")
