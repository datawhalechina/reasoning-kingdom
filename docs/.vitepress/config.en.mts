import { defineConfig } from 'vitepress'
import container from 'markdown-it-container'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/reasoning-kingdom/'

export default defineConfig({
  lang: 'en',
  title: "Reasoning Kingdom",
  description: "An open-source tutorial on AI reasoning mechanisms",
  base: baseConfig,
  appearance: false,

  markdown: {
    math: true,
    vue: { enabled: false },
    config: (md) => {
      const defaultRender = md.renderer.rules.html_inline || function(tokens, idx, options, env, self) {
        return self.renderToken(tokens, idx, options)
      }

      md.renderer.rules.html_inline = function(tokens, idx, options, env, self) {
        const token = tokens[idx]
        if (token.content.startsWith('<') && token.content.endsWith('>')) {
          const escaped = token.content
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
          return escaped
        }
        return defaultRender(tokens, idx, options, env, self)
      }

      md.use(container, 'info', {
        validate: function(params) { return params.trim() === 'info' },
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            return '<div class="info custom-block">\n'
          } else {
            return '</div>\n'
          }
        }
      })
      md.use(container, 'detail', {
        validate: function(params) { return params.trim() === 'detail' },
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            return '<div class="detail custom-block">\n'
          } else {
            return '</div>\n'
          }
        }
      })

      md.use(container, 'details', {
        validate: function(params) {
          return !!params.trim().match(/^details(::|\s|$)/)
        },
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const info = tokens[idx].info.trim()
            const title = info.replace(/^details\s*:?\s*/, '')
            if (title) {
              return `<div class="details custom-block"><p class="details-summary">${title}</p>\n`
            }
            return '<div class="details custom-block">\n'
          } else {
            return '</div>\n'
          }
        }
      })
    }
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: 'Map', link: '/map' },
      { text: 'Prequel: Intro to Reasoning Science', link: '/dear-reasoner/preface' },
      { text: 'Volume I: Reasoning Narratives', link: '/volume1/preface/' },
      { text: 'Volume II: Formal Deduction', link: '/volume2/preface/' },
      { text: 'Pallas Academy', link: '/dear-reasoner/academy/' },
      { text: 'Dictionary', link: '/dictionary' },
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
          text: 'Prequel: Introduction to Reasoning Science',
          items: [
            { text: 'Preface', link: '/dear-reasoner/preface' },
            {
              text: 'Part I: The Universe of Certainty',
              items: [
                { text: 'Ch1: Telegraph, Flashlight, and the Origin of Logic', link: '/dear-reasoner/volume1/chapter1/' },
                { text: 'Ch2: When Resources Have Boundaries (Complexity)', link: '/dear-reasoner/volume1/chapter2/' },
                { text: 'Ch3: Turing\'s Paper Tape (Computability)', link: '/dear-reasoner/volume1/chapter3/' },
                { text: 'Ch4: The Wisdom of Linearity (Traversal & Search)', link: '/dear-reasoner/volume1/chapter4/' },
                { text: 'Ch5: The Temptation of Greed (Local Optima)', link: '/dear-reasoner/volume1/chapter5/' },
                { text: 'Ch6: The Art of Heuristics (Approximation & Estimation)', link: '/dear-reasoner/volume1/chapter6/' },
                { text: 'Ch7: The Power of Memory (Dynamic Programming)', link: '/dear-reasoner/volume1/chapter7/' },
              ]
            },
            {
              text: 'Part II: Crossing the Fracture Zone of Logic',
              items: [
                { text: 'Ch8: The Twilight of Rules', link: '/dear-reasoner/volume2/chapter8/' },
                { text: 'Ch9: From Discrete to Continuous', link: '/dear-reasoner/volume2/chapter9/' },
              ]
            },
            {
              text: 'Part III: The Emergence of Neural Networks',
              items: [
                { text: 'Ch10: The Simplest Perception (The Neuron)', link: '/dear-reasoner/volume3/chapter10/' },
                { text: 'Ch11: Error Is the Ladder of Progress (Backpropagation)', link: '/dear-reasoner/volume3/chapter11/' },
                { text: 'Ch12: Chains of Memory (LSTM and RNN)', link: '/dear-reasoner/volume3/chapter12/' },
                { text: 'Ch13: The Contest Between Forgetting and Causality', link: '/dear-reasoner/volume3/chapter13/' },
                { text: 'Ch14: Attention: Where Should We Look?', link: '/dear-reasoner/volume3/chapter14/' },
                { text: 'Ch15: The Encoder-Decoder Stack (Transformer)', link: '/dear-reasoner/volume3/chapter15/' },
              ]
            },
            {
              text: 'Part IV: The Path to the Reasoning Kingdom',
              items: [
                { text: 'Ch16: What Is True Reasoning? (The LLM Myth)', link: '/dear-reasoner/volume4/chapter16/' },
                { text: 'Ch17: The Reasoning Scientist\'s Toolbox', link: '/dear-reasoner/volume4/chapter17/' },
                { text: 'Ch18: To You Beyond Age 20: Reasoning as a Science', link: '/dear-reasoner/volume4/chapter18/' },
              ]
            },
          ]
        }
      ],
       '/volume1/': [
        {
          text: 'Volume I: Historical Narratives of Reasoning',
           items: [
            { text: 'Preface', link: '/volume1/preface/' },
            { text: 'Ch1: Against Entropy — Reasoning as a Survival Strategy', link: '/volume1/chapter1/' },
            { text: 'Ch2: The Dawn of Symbols — The First Modeling of Causality', link: '/volume1/chapter2/' },
            { text: 'Ch3: From Symbols to Vectors — The First Liberation of Representation Space', link: '/volume1/chapter3/' },
            { text: 'Ch4: The Manifold Hypothesis — The Hidden Order of High-Dimensional Data', link: '/volume1/chapter4/' },
            { text: 'Ch5: The Trap of Fitting — Statistical Correlation Is Not Reasoning', link: '/volume1/chapter5/' },
            { text: 'Ch6: The Boundaries of Causality — Observational Data Is Never Enough', link: '/volume1/chapter6/' },
            { text: 'Ch7: The Truth About Complexity: It\'s About Structure, Not Speed', link: '/volume1/chapter7/' },
            { text: 'Ch8: The Contract of Heuristics: How Much Courage Does It Take to Accept "Close Enough"?', link: '/volume1/chapter8/' },
            { text: 'Ch9: Transformer: The Attention Revolution of Dynamic Topology', link: '/volume1/chapter9/' },
            { text: '↳ Bonus: Attention Is Causality', link: '/volume1/chapter9/bonus' },
            { text: 'Ch10: The Art of Search: Cruising Through Reasoning Space', link: '/volume1/chapter10/' },
            { text: 'Ch11: Efficient Reasoning: The Economics of Algorithms', link: '/volume1/chapter11/' },
            { text: 'Ch12: Implicit Reasoning: The Neural Network\'s Internal Monologue', link: '/volume1/chapter12/' },
            { text: 'Ch13: The Boundaries of Reasoning — and Why We Must Accept Them', link: '/volume1/chapter13/' },
            { text: '↳ Bonus: The Hidden Thread', link: '/volume1/chapter13/bonus' },
            { text: 'Bonus: CocDo — Neural Causal Operators', link: '/volume1/chapterbonous/' },
            { text: 'Dictionary', link: '/dictionary' },
          ]
        }
      ],
       '/volume2/': [
        {
          text: 'Volume II: Formal Deduction of Reasoning',
          items: [
            { text: 'Preface: Before Building on the Foundation', link: '/volume2/preface/' },
            { text: 'Ch14: Formal Systems — Giving Reasoning a Foundation', link: '/volume2/chapter14/' },
            { text: 'Ch15: Consistency and Completeness — The Two Walls of Formal Systems', link: '/volume2/chapter15/' },
            { text: 'Ch16: Linear Logic and Resources — Every Hypothesis Can Be Used Only Once', link: '/volume2/chapter16/' },
            { text: 'Ch17: Probability as the Expansion of Logic — Truth Values from {0,1} to [0,1]', link: '/volume2/chapter17/' },
            { text: 'Ch18: Formalizing Causal Structure — The Three-Rung Ladder and do-Calculus', link: '/volume2/chapter18/' },
            { text: 'Ch19: Complexity as the Geometry of Reasoning — Why Some Reasoning Cannot Be Accelerated', link: '/volume2/chapter19/' },
            { text: 'Ch20: The Formal Contract of Heuristics — The Precise Mathematical Definition of "Approximately Right"', link: '/volume2/chapter20/' },
            { text: 'Ch21: Learning as Inverse Inference — Generalization Is Compression by Another Name', link: '/volume2/chapter21/' },
            { text: 'Ch22: Self-Reference and Emergence — When a Reasoning System Begins to Reason About Itself', link: '/volume2/chapter22/' },
            { text: 'Ch23: Yonglin-Lyapunov Correspondence — Stability and Convergence Boundaries of Reasoning Systems', link: '/volume2/chapter23/' },
            { text: 'Ch24: Inference Convergence Through the Lens of Category Theory — Ghost Pointers, Terminal Objects, and Adjoint Functors', link: '/volume2/chapter24/' },
            { text: 'Dictionary', link: '/dictionary' },
          ]
        }
      ],
        '/': [
        {
            items: [
             { text: 'Reasoning Kingdom Map', link: '/map' },
             { text: 'Preface', link: '/volume1/preface/' },
             { text: 'Prequel: Introduction to Reasoning Science →', link: '/dear-reasoner/preface' },
             { text: 'Dictionary', link: '/dictionary' },
            { text: 'Volume I: Historical Narratives →', link: '/volume1/preface/' },
            { text: 'Volume II: Formal Deduction →', link: '/volume2/preface/' },
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
      copyright: 'Licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">CC BY-NC-SA 4.0</a>'
    }
  }
})
