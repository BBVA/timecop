<template>
  <v-flex xs12 v-resize="calculateSize">
    <v-toolbar dense>
      <v-toolbar-items>
        <v-btn
          v-for="(g, i) in toGraph"
          :key="'btn' + i"
          :color="g.color"
          :disabled="extendedArea.active"
          :class="{'graph-inactive-btn': !g.visible}"
          @click="g.visible = !g.visible" flat>
          {{g.name}}
          <span v-if="g.name === dataSet.prediction.engine" class="winner">&#9813;</span>
        </v-btn>
      </v-toolbar-items>
      <v-spacer></v-spacer>
      <v-tooltip bottom>
        <v-btn :disabled="extendedArea.active" slot="activator" flat @click="randomizeColors" icon>
          <v-icon>brush</v-icon>
        </v-btn>
        <span>Randomize Colors</span>
      </v-tooltip>
      <v-tooltip bottom>
        <v-btn :disabled="total.length === 0" slot="activator" flat @click="extendedArea.active = !extendedArea.active" icon>
          <v-icon>crop_free</v-icon>
        </v-btn>
        <span>extend area</span>
      </v-tooltip>
    </v-toolbar>
    <svg width="100%" :height="height" :class="background"
    @wheel.prevent="zoom"
    @mousedown.exact="panEnabled = true"
    @mouseup="panEnabled = false"
    @mousemove="pan"
    ref="graph-container">
      <g :transform="`translate(${marginLeft}, ${marginTop})`"
       v-if="total.length > 0">
        <defs>
          <clipPath id="clip-rect">
            <rect :width="chartWidth" :height="height" :y="-this.marginTop" />
          </clipPath>
        </defs>
        <g clip-path="url(#clip-rect)">
        <!-- lineas de debug -->
        <g v-for="(g, i) in toGraph"
          :key="i"
          v-if="g.visible">
          <c-path
            :dasharray="g.debug ? '5,5' : ''"
            :transform="`translate(${- offsetX}, 0)`"
            :color="g.color"
            :rangeX="[zoomMin, zoomMax]"
            :rangeY="[globalMin, globalMax]"
            :dataset="g.data"
            y="y"
            x="x"
            :height="chartHeight"
            :width="chartWidth"></c-path>
        </g>

        <!-- marcas de anomalias -->
        <line
        v-if="markPos.pos > 0"
        :x1="markPos.pos"
        :x2="markPos.pos"
        :y2="chartHeight"
        stroke-width="2"
        stroke="#0eff0e78"
        fill="none"/>
        <text fill="#0eff0e78"
        v-if="markPos.pos > 0"
        text-anchor="middle"
        dy="-5px"
        :transform="`translate(${markPos.pos} 0)`">
          {{markPos.val}}
        </text>
        <circle
        v-for="(line, i) in anomalies"
        :key="i"
        :cx="line - offsetX" :cy="chartHeight" r="7" stroke="white" stroke-width="1" fill="red" />
        <!-- ejes -->
        <c-axis-x
        :transform="`translate(${-offsetX} ${chartHeight})`"
        :range="[zoomMin, zoomMax]"
        :dataset="total"
        x="x"
        :ticks="25"
        :fixed="1"
        :height="chartHeight"
        :width="chartWidth"
        :strokeColor="this.$vuetify.dark ? 'white' : '#6d6d6d'"></c-axis-x>
        </g>
        <c-axis-y
        :transform="`translate(${chartWidth - marginLeft - marginRight} 0)`"
        :range="[globalMin, globalMax]"
        :ticks="5"
        :fixed="3"
        :height="chartHeight"
        :strokeColor="this.$vuetify.dark ? 'white' : '#6d6d6d'"></c-axis-y>
      </g>
      <g v-if="extendedArea.active" :transform="`translate(${extendedArea.el.x}, ${extendedArea.el.y})`">
        <!-- center region -->
        <rect fill="#ffffff17" :width="extendedArea.el.w" :height="extendedArea.el.h"
        @mousemove="moveExtendedArea"
        @mousedown.exact="extendedArea.el.draggable = true"
        @mouseup="extendedArea.el.draggable = false"
        @mouseout="extendedArea.el.draggable = false"></rect>
        <!-- ctrl right -->
        <circle
          :cx="extendedArea.el.w"
          :cy="extendedArea.el.h / 2"
          r="15" stroke="black" fill="grey"
          @mousemove="extendedArea.el.w += $event.movementX" />
          <!-- ctrl bottom -->
        <circle
          :cx="extendedArea.el.w / 2"
          :cy="extendedArea.el.h"
          r="15" stroke="black" fill="grey"
          @mousemove="extendedArea.el.h += $event.movementY" />
      </g>
    </svg>
    <svg :viewBox="`${extendedArea.el.x} ${extendedArea.el.y} ${extendedArea.el.w} ${extendedArea.el.h}`"
    :class="background"
    width="100%"
    height="300"
    preserveAspectRatio="xMidYMid slice"
    v-html="extendedArea.value"></svg>
  </v-flex>
</template>

<script>
export default {
  props: {
    dataSet: null,
    height: {
      type: Number,
      default: 500
    },
    marginLeft: {
      type: Number,
      default: 40
    },
    marginTop: {
      type: Number,
      default: 40
    },
    marginRight: {
      type: Number,
      default: 20
    },
    marginBottom: {
      type: Number,
      default: 30
    },
    toggleSize: {
      type: Boolean
    },
    background: String,
    triggerReset: Boolean
  },
  data: () => ({
    width: null,
    chartWidth: 1000,
    chartHeight: 400,
    zoomMin: 0,
    zoomMax: 100,
    panEnabled: false,
    offsetX: 0,
    markPos: {
      pos: 0,
      val: 0
    },
    toGraph: {},
    total: [],
    extendedArea: {
      active: false,
      ready: false,
      value: '',
      el: {
        x: 0,
        y: 0,
        w: 500,
        h: 230,
        draggable: false,
        ctrlRigth: false,
        ctrlBottom: false
      }
    }
  }),
  mounted () {
    this.calculateSize()
  },
  methods: {
    calculateSize () {
      this.width = this.$el.clientWidth
      this.chartWidth = this.width - (this.marginLeft + this.marginRight)
      this.chartHeight = this.height - (this.marginTop + this.marginBottom)
    },
    drawData (d) {
      // to predict
      // take multivariate (main) or univariate (data)
      let toPredict = d.toPredict.main || d.toPredict.data
      this.$set(this.toGraph, 'main', {
        data: toPredict.map((v, i) => ({
          x: i,
          y: +v
        })),
        visible: true,
        color: this.resolveColor(this.toGraph.main),
        name: 'main'
      })
      // check for timeseries
      const timeseries = d.toPredict.timeseries
      if (timeseries && timeseries.length) {
        for (let t = 0; t < timeseries.length; t++) {
          this.$set(this.toGraph, 'data-' + t, {
            data: timeseries[t].data.map((v, i) => ({ x: i, y: +v })),
            visible: true,
            color: this.resolveColor(this.toGraph['data-' + t]),
            name: 'data-' + t
          })
        }
      }
      // debug engines
      if (d.prediction) {
        const res = d.prediction.status || d.prediction
        for (const key in res) {
          // no deberia hacer esto :/
          if (key === 'Holtwinters' || key === 'LSTM' || key === 'VAR' || key === 'Autoarima') {
            this.addDebugEngine(res[key], key)
          }
        }
      }

      // prediction
      const prediction = d.prediction.future
      if (prediction) {
        this.$set(this.toGraph, 'prediction', {
          data: prediction.map((v, i) => ({
            x: +v.step,
            y: +v['expected value'] || +v['value'] || +v['valores'] || +v['var_0'] || +v['values']
          })),
          visible: true,
          color: this.resolveColor(this.toGraph.prediction),
          name: 'prediction'
        })
        this.toGraph.prediction.data.unshift({
          x: this.toGraph.main.data[this.toGraph.main.data.length - 1].x,
          y: this.toGraph.main.data[this.toGraph.main.data.length - 1].y
        })
        // set default zoomMax width the length of main and prediction
        this.zoomMax += prediction.length
        this.total = this.total.concat(
          this.toGraph.prediction.data
        )
      }
      // set default zoomMax width the length of main and prediction
      if (this.zoomMax === 100) {
        this.zoomMax = toPredict.length
      }
      this.total = this.toGraph.main.data
    },
    zoom (e) {
      if (!this.extendedArea.active) {
        const delta = e.wheelDelta ? e.wheelDelta * 0.02 : -e.deltaY
        this.zoomMin += delta
        this.zoomMax += -delta
      }
    },
    pan (e) {
      if (this.panEnabled && !this.extendedArea.active) {
        this.offsetX += -e.movementX
        this.markPos = 0
      } else if (this.total.length > 0) {
        const position = e.offsetX - this.marginLeft
        for (let i = 0; i < this.total.length; i++) {
          let p = this.$utils.scale(i, this.zoomMin, this.zoomMax, this.chartWidth) - this.offsetX
          if (p < position) {
            this.markPos = {
              pos: p,
              val: i
            }
          }
        }
      }
    },
    addDebugEngine (engine, name) {
      this.$set(this.toGraph, name, {
        data: engine.debug.map(v => (
          {
            x: +v.step,
            y: +v['expected value'] ||
              +v['valores'] ||
              +v['var_0'] ||
              +v['values'] ||
              +v['value'] ||
              +v['Prediction']
          })
        ),
        visible: true,
        color: this.resolveColor(this.toGraph[name]),
        name: name,
        debug: true
      })
    },
    moveExtendedArea (e) {
      if (this.extendedArea.el.draggable) {
        this.extendedArea.el.x += e.movementX
        this.extendedArea.el.y += e.movementY
      }
    },
    resolveColor (expression) {
      if (expression && expression.color) {
        return expression.color
      } else {
        return this.$utils.getRandomColor()
      }
    },
    randomizeColors () {
      for (const graph in this.toGraph) {
        this.toGraph[graph].color = this.$utils.getRandomColor()
      }
      this.$nextTick(() => {
        this.extendedArea.value = this.$refs['graph-container'].innerHTML
      })
    },
    reset () {
      this.zoomMin = 0
      this.zoomMax = 100
      this.offsetX = 0
      this.toGraph = {}
      this.extendedArea.active = false
    }
  },
  watch: {
    triggerReset: function () {
      this.reset()
    },
    dataSet: {
      handler: function (val) {
        if (val.toPredict && val.prediction) {
          this.$nextTick(() => {
            this.drawData(val)
            this.$nextTick(() => {
              this.extendedArea.value = this.$refs['graph-container'].innerHTML
              this.extendedArea.ready = true
            })
          })
        }
      },
      immediate: true
    },
    toggleSize: function () {
      this.calculateSize()
    },
    extendedAreaState: function (val) {
      if (val) {
        this.extendedArea.value = this.$refs['graph-container'].innerHTML
      }
    }
  },
  computed: {
    anomalies () {
      const d = this.dataSet
      if (d.toPredict && d.prediction.past) {
        let anomalies = []
        for (let i = 0; i < d.prediction.past.length; i++) {
          const point = d.prediction.past[i].step
          if (point) {
            let xMark = this.$utils.scale(d.prediction.past[i].step, this.zoomMin, this.zoomMax, this.chartWidth)
            anomalies.push(xMark)
          }
        }
        return anomalies
      }
      return []
    },
    globalMax () {
      let globalMax = -10e10
      for (const v in this.toGraph) {
        if (this.toGraph[v].visible) {
          const localMax = this.$utils.getMax(this.toGraph[v].data, 'y')
          globalMax = globalMax > localMax ? globalMax : localMax
        }
      }
      return globalMax
    },
    globalMin () {
      let globalMin = 10e10
      for (const v in this.toGraph) {
        if (this.toGraph[v].visible) {
          const localMin = this.$utils.getMin(this.toGraph[v].data, 'y')
          globalMin = globalMin < localMin ? globalMin : localMin
        }
      }
      return globalMin
    },
    extendedAreaState () {
      return this.extendedArea.active
    }
  }
}
</script>

<style>
.winner {
  position: absolute;
  right: 0;
  margin-right: -14px;
  margin-top: -10px;
  font-size: 25px;
  transform: rotateZ(28deg);
}
.graph-inactive-btn {
  text-decoration: line-through !important;
  text-decoration-style: double !important;
}
.extended-area {
  width: 100%;
  height: 250px;
}
</style>
