import { defineConfig } from 'vitepress'
import container from 'markdown-it-container'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/reasoning-kingdom/'

const mdConfig = (md: any) => {
  const defaultRender = md.renderer.rules.html_inline || function(tokens: any, idx: any, options: any, env: any, self: any) {
    return self.renderToken(tokens, idx, options)
  }
  md.renderer.rules.html_inline = function(tokens: any, idx: any, options: any, env: any, self: any) {
    const token = tokens[idx]
    if (token.content.startsWith('<') && token.content.endsWith('>')) {
      return token.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')
    }
    return defaultRender(tokens, idx, options, env, self)
  }
  md.use(container, 'info', {
    validate: (p: string) => p.trim() === 'info',
    render: (tokens: any[], idx: number) => tokens[idx].nesting === 1 ? '<div class="info custom-block">\n' : '</div>\n'
  })
  md.use(container, 'detail', {
    validate: (p: string) => p.trim() === 'detail',
    render: (tokens: any[], idx: number) => tokens[idx].nesting === 1 ? '<div class="detail custom-block">\n' : '</div>\n'
  })
  md.use(container, 'details', {
    validate: (p: string) => !!p.trim().match(/^details(::|\s|$)/),
    render: (tokens: any[], idx: number) => {
      if (tokens[idx].nesting === 1) {
        const title = tokens[idx].info.trim().replace(/^details\s*:?\s*/, '')
        return title
          ? `<div class="details custom-block"><p class="details-summary">${title}</p>\n`
          : '<div class="details custom-block">\n'
      }
      return '</div>\n'
    }
  })
}

export default defineConfig({
  base: baseConfig,
  appearance: false,

  locales: {
    root: {
      label: '中文',
      lang: 'zh-CN',
      title: '推理王国',
      description: '一本关于AI推理机制的开源教程',
      themeConfig: {
        logo: '/datawhale-logo.png',
        nav: [
          { text: '推理地图', link: '/map' },
          { text: '前传：推理科学入门', link: '/dear-reasoner/preface' },
          { text: '上卷：推理叙事', link: '/volume1/preface/' },
          { text: '下卷：形式演绎', link: '/volume2/preface/' },
          { text: '兔狲学院', link: '/dear-reasoner/academy/' },
          { text: '兔狲教授小词典', link: '/dictionary' },
          { text: 'English →', link: '/en/' },
        ],
        search: {
          provider: 'local',
          shortcut: {
            search: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' },
            open: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' }
          }
        },
        sidebar: {
          '/dear-reasoner/': [
            {
              text: '前传：推理科学入门',
              items: [
                { text: '前言', link: '/dear-reasoner/preface' },
                {
                  text: '第一部分：确定性的宇宙',
                  items: [
                    { text: '第1章：电报、手电筒与逻辑的起点', link: '/dear-reasoner/volume1/chapter1/' },
                    { text: '第2章：当资源有了边界（复杂度）', link: '/dear-reasoner/volume1/chapter2/' },
                    { text: '第3章：图灵的纸带（可计算性）', link: '/dear-reasoner/volume1/chapter3/' },
                    { text: '第4章：线性的智慧（遍历与搜索）', link: '/dear-reasoner/volume1/chapter4/' },
                    { text: '第5章：贪心的诱惑（局部最优）', link: '/dear-reasoner/volume1/chapter5/' },
                    { text: '第6章：启发的艺术（近似与估测）', link: '/dear-reasoner/volume1/chapter6/' },
                    { text: '第7章：记忆的力量（动态规划）', link: '/dear-reasoner/volume1/chapter7/' },
                  ]
                },
                {
                  text: '第二部分：跨越逻辑的断裂带',
                  items: [
                    { text: '第8章：规则的黄昏', link: '/dear-reasoner/volume2/chapter8/' },
                    { text: '第9章：从离散到连续：数学的魔法棒', link: '/dear-reasoner/volume2/chapter9/' },
                  ]
                },
                {
                  text: '第三部分：神经网络的涌现',
                  items: [
                    { text: '第10章：最简单的感知（神经元）', link: '/dear-reasoner/volume3/chapter10/' },
                    { text: '第11章：错误是进步的阶梯（反向传播）', link: '/dear-reasoner/volume3/chapter11/' },
                    { text: '第12章：记忆的链条（LSTM与RNN）', link: '/dear-reasoner/volume3/chapter12/' },
                    { text: '第13章：遗忘与因果的较量', link: '/dear-reasoner/volume3/chapter13/' },
                    { text: '第14章：注意力：在这个嘈杂的世界里，该看哪？', link: '/dear-reasoner/volume3/chapter14/' },
                    { text: '第15章：编码-解码堆栈（Transformer）', link: '/dear-reasoner/volume3/chapter15/' },
                  ]
                },
                {
                  text: '第四部分：通往推理王国',
                  items: [
                    { text: '第16章：什么是真正的推理？（LLM迷思）', link: '/dear-reasoner/volume4/chapter16/' },
                    { text: '第17章：推理科学家的工具箱', link: '/dear-reasoner/volume4/chapter17/' },
                    { text: '第18章：致20岁后的你：作为科学的推理', link: '/dear-reasoner/volume4/chapter18/' },
                  ]
                },
                {
                  text: '兔狲学院',
                  items: [
                    { text: '学院主页：推理科学家的思维实验室', link: '/dear-reasoner/academy/' },
                    {
                      text: '数学基础综合：从自然数到不动点理论',
                      link: '/dear-reasoner/academy/math-core/',
                      items: [
                        { text: '自然数与公理系统', link: '/dear-reasoner/academy/math-core/sections/natural-numbers.md' },
                        { text: '集合论基础', link: '/dear-reasoner/academy/math-core/sections/set-theory.md' },
                        { text: '逻辑与证明方法', link: '/dear-reasoner/academy/math-core/sections/logic.md' },
                        { text: '函数与关系', link: '/dear-reasoner/academy/math-core/sections/functions.md' },
                        { text: '数列与极限', link: '/dear-reasoner/academy/math-core/sections/sequences.md' },
                        { text: 'ZFC集合论', link: '/dear-reasoner/academy/math-core/sections/zfc.md' },
                        { text: '不动点理论', link: '/dear-reasoner/academy/math-core/sections/fixed-points.md' },
                      ]
                    },
                    {
                      text: 'AI数学基础：概率统计到线性模型',
                      link: '/dear-reasoner/academy/ai-mathematics/',
                      items: [
                        { text: '概率论基础', link: '/dear-reasoner/academy/ai-mathematics/sections/probability' },
                        { text: '统计学基础', link: '/dear-reasoner/academy/ai-mathematics/sections/statistics' },
                        { text: '优化理论', link: '/dear-reasoner/academy/ai-mathematics/sections/optimization' },
                        { text: '信息论', link: '/dear-reasoner/academy/ai-mathematics/sections/information-theory' },
                        { text: '线性模型', link: '/dear-reasoner/academy/ai-mathematics/sections/linear-models' },
                      ]
                    },
                    {
                      text: '哲学：从古希腊到1840年',
                      link: '/dear-reasoner/academy/philosophy/',
                      items: [
                        { text: '古希腊哲学', link: '/dear-reasoner/academy/philosophy/sections/ancient-greek.md' },
                        { text: '文艺复兴哲学', link: '/dear-reasoner/academy/philosophy/sections/renaissance.md' },
                        { text: '德国古典哲学', link: '/dear-reasoner/academy/philosophy/sections/german-idealism.md' },
                        { text: '马克思与恩格斯', link: '/dear-reasoner/academy/philosophy/sections/marx-engels.md' },
                        { text: '企望新哲学', link: '/dear-reasoner/academy/philosophy/sections/future.md' },
                      ]
                    },
                    {
                      text: 'Python编程：从语法到数据结构',
                      link: '/dear-reasoner/academy/python/',
                      items: [
                        { text: 'Python基础语法', link: '/dear-reasoner/academy/python/sections/basics.md' },
                        { text: '数据结构', link: '/dear-reasoner/academy/python/sections/data_structures.md' },
                        { text: '算法设计', link: '/dear-reasoner/academy/python/sections/algorithms.md' },
                        { text: '高级应用', link: '/dear-reasoner/academy/python/sections/advanced.md' },
                      ]
                    },
                    { text: '微积分：从函数到微分方程', link: '/dear-reasoner/academy/calculus/' },
                    { text: '线性代数：从向量到雅可比矩阵', link: '/dear-reasoner/academy/linear-algebra/' },
                  ]
                }
              ]
            }
          ],
          '/volume1/': [
            {
              text: '上卷：推理的历史叙事',
              items: [
                { text: '导读', link: '/volume1/preface/' },
                { text: '第1章：对抗熵增——推理作为存活策略', link: '/volume1/chapter1/' },
                { text: '第2章：符号的黎明——因果的第一次建模', link: '/volume1/chapter2/' },
                { text: '第3章：从符号到向量——表示空间的第一次解放', link: '/volume1/chapter3/' },
                { text: '第4章：流形假设——高维数据的隐秩序', link: '/volume1/chapter4/' },
                { text: '第5章：拟合的陷阱——统计相关性不是推理', link: '/volume1/chapter5/' },
                { text: '第6章：因果的边界——观测数据永远不够', link: '/volume1/chapter6/' },
                { text: '第7章：复杂度的真相：不是快慢，是结构', link: '/volume1/chapter7/' },
                { text: '第8章：启发式的契约：接受"差不多对"需要多少勇气', link: '/volume1/chapter8/' },
                { text: '第9章：Transformer：动态拓扑的注意力革命', link: '/volume1/chapter9/' },
                { text: '↳ 番外篇：注意力即因果', link: '/volume1/chapter9/bonus' },
                { text: '第10章：搜索的艺术：在推理空间中巡航', link: '/volume1/chapter10/' },
                { text: '第11章：效能化推理：算法的经济学', link: '/volume1/chapter11/' },
                { text: '第12章：隐式推理：神经网络的内部独白', link: '/volume1/chapter12/' },
                { text: '第13章：推理的边界——以及我们为什么必须接受它', link: '/volume1/chapter13/' },
                { text: '↳ 番外篇：暗线', link: '/volume1/chapter13/bonus' },
                { text: '因果推理番外篇：CocDo 神经因果算子', link: '/volume1/chapterbonous/' },
                { text: '兔狲教授小词典', link: '/dictionary' },
              ]
            }
          ],
          '/volume2/': [
            {
              text: '下卷：推理的形式演绎',
              items: [
                { text: '下卷导读：在地基上建造之前', link: '/volume2/preface/' },
                { text: '第14章：形式系统——给推理一个地基', link: '/volume2/chapter14/' },
                { text: '第15章：一致性与完备性——形式系统的两堵墙', link: '/volume2/chapter15/' },
                { text: '第16章：线性逻辑与资源——每个假设只能用一次', link: '/volume2/chapter16/' },
                { text: '第17章：概率作为逻辑的扩张——真值从 {0,1} 到 [0,1]', link: '/volume2/chapter17/' },
                { text: '第18章：因果结构的形式化——三层阶梯与 do-calculus', link: '/volume2/chapter18/' },
                { text: '第19章：复杂度作为推理的几何——为什么有些推理根本不能被加速', link: '/volume2/chapter19/' },
                { text: '第20章：启发式的形式合同——"差不多对"的精确数学定义', link: '/volume2/chapter20/' },
                { text: '第21章：学习作为逆推断——泛化是压缩的另一种说法', link: '/volume2/chapter21/' },
                { text: '第22章：自指与涌现——当推理系统开始推理关于自身', link: '/volume2/chapter22/' },
                { text: '第23章：永霖-李雅普诺夫联立——推理系统的稳定性与收敛边界', link: '/volume2/chapter23/' },
                { text: '第24章：范畴论眼中的推理收敛——链表、指针与伴随函子', link: '/volume2/chapter24/' },
                { text: '第25章：边界的统一——八条界线与一个不可能三角', link: '/volume2/chapter25/' },
                { text: '兔狲教授小词典', link: '/dictionary' },
                { text: '附录：下卷思考题参考提示', link: '/appendix-thinking-questions' },
              ]
            }
          ],
          '/': [
            {
              items: [
                { text: '推理王国地图', link: '/map' },
                { text: '导读', link: '/volume1/preface/' },
                { text: '前传：推理科学入门 →', link: '/dear-reasoner/preface' },
                { text: '兔狲教授小词典', link: '/dictionary' },
                { text: '上卷：推理的历史叙事 →', link: '/volume1/preface/' },
                { text: '下卷：推理的形式演绎 →', link: '/volume2/preface/' },
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/reasoning-kingdom/' }
        ],
        editLink: {
          pattern: 'https://github.com/datawhalechina/reasoning-kingdom/blob/main/docs/:path'
        },
        footer: {
          message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
          copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en',
      title: 'Reasoning Kingdom',
      description: 'An open-source tutorial on AI reasoning mechanisms',
      link: '/en/',
      themeConfig: {
        logo: '/datawhale-logo.png',
        nav: [
          { text: '中文 →', link: '/' },
          { text: 'Prequel', link: '/en/dear-reasoner/preface' },
          { text: 'Volume I', link: '/en/volume1/preface/' },
          { text: 'Volume II', link: '/en/volume2/preface/' },
        ],
        search: {
          provider: 'local',
          shortcut: {
            search: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' },
            open: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' }
          }
        },
        sidebar: {
          '/en/dear-reasoner/': [
            {
              text: 'Prequel: Introduction to Reasoning Science',
              items: [
                { text: 'Preface', link: '/en/dear-reasoner/preface' },
                {
                  text: 'Part I: The Universe of Certainty',
                  items: [
                    { text: 'Ch1: Telegraph, Flashlight, and the Origin of Logic', link: '/en/dear-reasoner/volume1/chapter1/' },
                    { text: 'Ch2: When Resources Have Boundaries (Complexity)', link: '/en/dear-reasoner/volume1/chapter2/' },
                    { text: "Ch3: Turing's Paper Tape (Computability)", link: '/en/dear-reasoner/volume1/chapter3/' },
                    { text: 'Ch4: The Wisdom of Linearity (Traversal & Search)', link: '/en/dear-reasoner/volume1/chapter4/' },
                    { text: 'Ch5: The Temptation of Greed (Local Optima)', link: '/en/dear-reasoner/volume1/chapter5/' },
                    { text: 'Ch6: The Art of Heuristics (Approximation & Estimation)', link: '/en/dear-reasoner/volume1/chapter6/' },
                    { text: 'Ch7: The Power of Memory (Dynamic Programming)', link: '/en/dear-reasoner/volume1/chapter7/' },
                  ]
                },
                {
                  text: 'Part II: Crossing the Fracture Zone of Logic',
                  items: [
                    { text: 'Ch8: The Twilight of Rules', link: '/en/dear-reasoner/volume2/chapter8/' },
                    { text: 'Ch9: From Discrete to Continuous', link: '/en/dear-reasoner/volume2/chapter9/' },
                  ]
                },
                {
                  text: 'Part III: The Emergence of Neural Networks',
                  items: [
                    { text: 'Ch10: The Simplest Perception (The Neuron)', link: '/en/dear-reasoner/volume3/chapter10/' },
                    { text: 'Ch11: Error Is the Ladder of Progress (Backpropagation)', link: '/en/dear-reasoner/volume3/chapter11/' },
                    { text: 'Ch12: Chains of Memory (LSTM and RNN)', link: '/en/dear-reasoner/volume3/chapter12/' },
                    { text: 'Ch13: The Contest Between Forgetting and Causality', link: '/en/dear-reasoner/volume3/chapter13/' },
                    { text: 'Ch14: Attention: Where Should We Look?', link: '/en/dear-reasoner/volume3/chapter14/' },
                    { text: 'Ch15: The Encoder-Decoder Stack (Transformer)', link: '/en/dear-reasoner/volume3/chapter15/' },
                  ]
                },
                {
                  text: 'Part IV: The Path to the Reasoning Kingdom',
                  items: [
                    { text: 'Ch16: What Is True Reasoning? (The LLM Myth)', link: '/en/dear-reasoner/volume4/chapter16/' },
                    { text: "Ch17: The Reasoning Scientist's Toolbox", link: '/en/dear-reasoner/volume4/chapter17/' },
                    { text: 'Ch18: To You Beyond Age 20: Reasoning as a Science', link: '/en/dear-reasoner/volume4/chapter18/' },
                  ]
                },
              ]
            }
          ],
          '/en/volume1/': [
            {
              text: 'Volume I: Historical Narratives of Reasoning',
              items: [
                { text: 'Preface', link: '/en/volume1/preface/' },
                { text: 'Ch1: Against Entropy — Reasoning as a Survival Strategy', link: '/en/volume1/chapter1/' },
                { text: 'Ch2: The Dawn of Symbols — The First Modeling of Causality', link: '/en/volume1/chapter2/' },
                { text: 'Ch3: From Symbols to Vectors — The First Liberation of Representation Space', link: '/en/volume1/chapter3/' },
                { text: 'Ch4: The Manifold Hypothesis — The Hidden Order of High-Dimensional Data', link: '/en/volume1/chapter4/' },
                { text: 'Ch5: The Trap of Fitting — Statistical Correlation Is Not Reasoning', link: '/en/volume1/chapter5/' },
                { text: 'Ch6: The Boundaries of Causality — Observational Data Is Never Enough', link: '/en/volume1/chapter6/' },
                { text: "Ch7: The Truth About Complexity: It's About Structure, Not Speed", link: '/en/volume1/chapter7/' },
                { text: 'Ch8: The Contract of Heuristics', link: '/en/volume1/chapter8/' },
                { text: 'Ch9: Transformer: The Attention Revolution of Dynamic Topology', link: '/en/volume1/chapter9/' },
                { text: '↳ Bonus: Attention Is Causality', link: '/en/volume1/chapter9/bonus' },
                { text: 'Ch10: The Art of Search: Cruising Through Reasoning Space', link: '/en/volume1/chapter10/' },
                { text: 'Ch11: Efficient Reasoning: The Economics of Algorithms', link: '/en/volume1/chapter11/' },
                { text: "Ch12: Implicit Reasoning: The Neural Network's Internal Monologue", link: '/en/volume1/chapter12/' },
                { text: 'Ch13: The Boundaries of Reasoning — and Why We Must Accept Them', link: '/en/volume1/chapter13/' },
                { text: '↳ Bonus: The Hidden Thread', link: '/en/volume1/chapter13/bonus' },
                { text: 'Bonus: CocDo — Neural Causal Operators', link: '/en/volume1/chapterbonous/' },
                { text: 'Dictionary', link: '/en/dictionary' },
              ]
            }
          ],
          '/en/volume2/': [
            {
              text: 'Volume II: Formal Deduction of Reasoning',
              items: [
                { text: 'Preface: Before Building on the Foundation', link: '/en/volume2/preface/' },
                { text: 'Ch14: Formal Systems — Giving Reasoning a Foundation', link: '/en/volume2/chapter14/' },
                { text: 'Ch15: Consistency and Completeness — The Two Walls of Formal Systems', link: '/en/volume2/chapter15/' },
                { text: 'Ch16: Linear Logic and Resources — Every Hypothesis Can Be Used Only Once', link: '/en/volume2/chapter16/' },
                { text: 'Ch17: Probability as the Expansion of Logic — Truth Values from {0,1} to [0,1]', link: '/en/volume2/chapter17/' },
                { text: 'Ch18: Formalizing Causal Structure — The Three-Rung Ladder and do-Calculus', link: '/en/volume2/chapter18/' },
                { text: 'Ch19: Complexity as the Geometry of Reasoning', link: '/en/volume2/chapter19/' },
                { text: 'Ch20: The Formal Contract of Heuristics', link: '/en/volume2/chapter20/' },
                { text: 'Ch21: Learning as Inverse Inference — Generalization Is Compression by Another Name', link: '/en/volume2/chapter21/' },
                { text: 'Ch22: Self-Reference and Emergence', link: '/en/volume2/chapter22/' },
                { text: 'Ch23: Yonglin-Lyapunov Correspondence', link: '/en/volume2/chapter23/' },
                { text: 'Ch24: Inference Convergence Through the Lens of Category Theory', link: '/en/volume2/chapter24/' },
                { text: 'Ch25: The Unification of Boundaries — Eight Lines and an Impossible Triangle', link: '/en/volume2/chapter25/' },
                { text: 'Dictionary', link: '/en/dictionary' },
              ]
            }
          ],
          '/en/': [
            {
              items: [
                { text: 'Preface', link: '/en/volume1/preface/' },
                { text: 'Prequel: Introduction to Reasoning Science →', link: '/en/dear-reasoner/preface' },
                { text: 'Volume I: Historical Narratives →', link: '/en/volume1/preface/' },
                { text: 'Volume II: Formal Deduction →', link: '/en/volume2/preface/' },
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/reasoning-kingdom/' }
        ],
        editLink: {
          pattern: 'https://github.com/datawhalechina/reasoning-kingdom/blob/main/docs/:path'
        },
        footer: {
          message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a>',
          copyright: 'Licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">CC BY-NC-SA 4.0</a>'
        }
      }
    }
  },

  markdown: {
    math: true,
    vue: { enabled: false },
    config: mdConfig
  }
})
