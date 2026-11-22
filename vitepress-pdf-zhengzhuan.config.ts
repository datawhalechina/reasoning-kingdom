import { defineUserConfig } from 'vitepress-export-pdf'

// 正传（上卷 volume1 + 下卷 volume2）按书籍顺序导出：
// 上卷导读 → 第1–13章（含番外）→ 下卷导读 → 第14–24章
function sortKey(path: string): [number, number, number] {
  const vol = path.includes('/volume1/') ? 1 : 2

  // 导读排在本卷最前
  if (/preface/.test(path)) return [vol, 0, 0]

  // 章号取路径中 chapter 后的数字（page.path 形如 /volume2/chapter23/index.html）
  const m = path.match(/chapter(\d+)/)
  const ch = m ? Number(m[1]) : 0

  // 同一章的番外/bonus 紧跟正文之后
  const extra = /bonus|bonous/i.test(path) ? 1 : 0

  return [vol, ch, extra]
}

export default defineUserConfig({
  routePatterns: [
    // 只保留上卷/下卷
    '!**/dear-reasoner/**',
    '!**/en/**',
    // 排除全站通用页
    '!/index.html',
    '!/reasoning-kingdom/index.html',
    '!/map.html',
    '!/reasoning-kingdom/map.html',
    '!/dictionary.html',
    '!/reasoning-kingdom/dictionary.html',
  ],
  sorter: (a, b) => {
    const ka = sortKey(a.path)
    const kb = sortKey(b.path)
    for (let i = 0; i < 3; i++) {
      if (ka[i] !== kb[i]) return ka[i] - kb[i]
    }
    return 0
  },
  puppeteerLaunchOptions: {
    // 使用系统 Chrome（未下载 puppeteer 自带 Chromium 时）
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  },
  // 关闭自动书签：该工具的合并环节会打乱章节顺序；
  // 关闭后按 sorter 顺序合并，保证书籍顺序正确。
  pdfOutlines: false,
  pdfOptions: {
    format: 'A4',
    printBackground: true,
    scale: 1,
    margin: { top: '15mm', bottom: '15mm', left: '16mm', right: '16mm' },
  },
  outFile: '推理王国·正传（上下卷）.pdf',
  outDir: '.',
})
