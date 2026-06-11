async function test() {
  const url = 'https://www.reddit.com/r/ask/s/bkGdsmLFPL/';
  try {
    const res = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(url)}`);
    const data = await res.json();
    const html = data.contents;
    
    // Look for <link rel="canonical" href="...">
    const canonicalMatch = html.match(/<link rel="canonical" href="(.*?)"/);
    if (canonicalMatch && canonicalMatch[1]) {
      console.log('Canonical URL:', canonicalMatch[1]);
    } else {
      console.log('No canonical found. Snippet:', html.substring(0, 500));
      // fallback look for sh.reddit.com or something
      const ogUrlMatch = html.match(/<meta property="og:url" content="(.*?)"/);
      if (ogUrlMatch && ogUrlMatch[1]) {
        console.log('OG URL:', ogUrlMatch[1]);
      }
    }
  } catch (e) {
    console.error(e);
  }
}

test();
