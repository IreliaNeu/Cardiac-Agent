// Requires Playwright and a Chromium executable; screenshots stay in memory.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({executablePath: process.env.BROWSER_PATH, headless: true});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', err => errors.push(String(err)));
  const cases = ['patient0020', 'patient0004', 'patient0042', 'patient0039', 'patient0033'];
  const results = [];
  try {
    for (const viewport of [{width:1440,height:1000}, {width:390,height:844}]) {
      await page.setViewportSize(viewport);
      await page.goto(process.argv[2]);
      for (const patient of cases) {
        await page.selectOption('#case-select', patient);
        await page.waitForFunction(() => [...document.images].every(im => im.complete && im.naturalWidth > 0));
        const state = await page.evaluate(() => ({
          images: document.images.length,
          overflow: document.documentElement.scrollWidth > innerWidth,
          hash: location.hash,
          title: document.querySelector('#case-title').textContent,
          cohort: document.querySelector('#cohort').textContent,
          text: document.body.textContent,
        }));
        if(state.images !== 6 || state.overflow || state.hash !== `#${patient}` || !state.title.includes(patient))
          throw new Error(JSON.stringify(state));
        if (/NaN|undefined/.test(state.text)) throw new Error('Missing numerical data');
        await page.uncheck('#show-reference');
        if(await page.locator('figure.reference:visible').count()) throw new Error('Reference toggle failed');
        await page.check('#show-reference');
        await page.locator('#roles summary').first().click();
        if(!await page.locator('#roles details').first().getAttribute('open') &&
           !await page.locator('#roles details').first().evaluate(el => el.open)) throw new Error('Role detail failed');
        const links = await page.locator('a').evaluateAll(nodes => nodes.map(n => n.href));
        for(const url of links) {
          const response = await page.request.get(url);
          if(!response.ok()) throw new Error(`${response.status()} ${url}`);
        }
        results.push({patient, viewport, images:state.images, overflow:state.overflow});
      }
    }
    if(errors.length) throw new Error(errors.join('\n'));
    await page.setViewportSize({width:1440,height:1000});
    await page.selectOption('#case-select','patient0042');
    await page.waitForFunction(() => [...document.images].every(im => im.complete && im.naturalWidth));
    await page.evaluate(() => scrollTo(0, 0));
    const screenshots = [(await page.screenshot({type:'jpeg', quality:50})).toString('base64')];
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(() => scrollTo(0, 0));
    screenshots.push((await page.screenshot({type:'jpeg', quality:50})).toString('base64'));
    process.stdout.write(JSON.stringify({passed:true,results,errors,screenshots}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
