<template>
  <v-flex xs12>
    <div v-for="(item, i) in items" :key="i"
    @click="item.type !== 'object' || toggle(item.data, i + 1, item.level + 1, item.open), item.open = !item.open"
    :style="{'margin-left': item.level * 10 + 'px', cursor: item.type === 'object' && item.data.length !== 0 ? 'pointer' : 'auto'}">
      <span :class="{'font-weight-bold': item.type === 'object' && item.data.length !== 0}">{{item.name}}:</span>
      <span :class="getColor(item.type)">{{item.type === 'object' && item.data.length !== 0 ? '{...}' : item.data}}</span>
    </div>
  </v-flex>
</template>

<script>
export default {
  name: 'jsonViewer',
  props: ['json'],
  data: () => ({
    items: [],
    hightLevel: 0
  }),
  methods: {
    addItems (obj, index, level) {
      for (const key in obj) {
        this.items.splice(index, 0, {
          name: key,
          type: typeof obj[key],
          data: obj[key],
          level: level,
          open: true
        })
      }
      this.hightLevel = level
    },
    toggle (obj, index, level, open) {
      if (!open) {
        this.deleteItems(obj, index, level)
      } else {
        this.addItems(obj, index, level)
      }
    },
    deleteItems (obj, index, level) {
      this.hightLeve--
      const itemsLevel = this.items.filter(d => d.level >= level).length
      const itemLength = Object.keys(obj).length
      let toDelete = level !== this.hightLevel ? itemsLevel : itemLength
      for (let i = 0; i < toDelete; i++) {
        this.items.splice(index, 1)
      }
    },
    getColor (val) {
      switch (val) {
        case 'object': return 'green--text darken-4'
        case 'string': return 'red--text darken-4'
        case 'number': return 'blue--text darken-4'
        case 'boolean': return 'indigo--text darken-4'
      }
    }
  },
  watch: {
    json: {
      handler: function (val) {
        this.items = []
        this.addItems(val, 0, 0)
        this.items.reverse()
      },
      immediate: true
    }
  }
}
</script>