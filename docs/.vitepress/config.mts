import { defineConfig } from 'vitepress'
// https://vitepress.dev/reference/site-config

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/reasoning-kingdom/'

export default defineConfig({
  lang: 'zh-CN',
  title: "推理王国",
  description: "一本关于AI推理机制的开源教程",
  base: baseConfig,
  markdown: {
    math: true
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      {
        items: [
          { text: '导读', link: '/preface' },
          { text: '第1章：对抗熵增——推理作为存活策略', link: '/chapter1/' },
          { text: '第2章：符号的黎明——因果的第一次建模', link: '/chapter2/' },
          { text: '第3章：从符号到向量——表示空间的第一次解放', link: '/chapter3/' },
          { text: '第4章：流形假设——高维数据的隐秩序', link: '/chapter4/' },
          { text: '第5章：拟合的陷阱——统计相关性不是推理', link: '/chapter5/' },
          { text: '第6章：因果的边界——观测数据永远不够', link: '/chapter6/' },
          { text: '第7章：复杂度的真相：不是快慢，是结构', link: '/chapter7/' },
          { text: '第8章：启发式的契约：接受"差不多对"需要多少勇气', link: '/chapter8/' },
          { text: '第9章：Transformer：动态拓扑的注意力革命', link: '/chapter9/' },
          { text: '第10章：搜索的艺术：在推理空间中巡航', link: '/chapter10/' },
          { text: '第11章：效能化推理：算法的经济学', link: '/chapter11/' },
          { text: '第12章：隐式推理：神经网络的内部独白', link: '/chapter12/' },
          { text: '第13章：推理的边界——以及我们为什么必须接受它', link: '/chapter13/' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/lizixi-0x2F/ReasoningKingdom' }
    ],

    editLink: {
      pattern: 'https://github.com/lizixi-0x2F/ReasoningKingdom/blob/main/docs/:path'
    },

    footer: {
      message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
    }
  }
})
