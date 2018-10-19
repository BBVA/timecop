<template>
  <div>
    <v-btn outline color="blue" @click="$refs.csvFile.click()">csv</v-btn>
    <input type="file" hidden ref="csvFile" accept=".csv, text/plain" @change="loadCSVFile">
    <v-dialog
      v-model="selectHeaderDialog.value"
      persistent
      max-width="500"
    >
      <v-card dark>
        <v-card-title>
          <div>
            <h3 class="headline">What column do you want to process?</h3>
            <small v-if="amountSelectedData > 0">data type: {{amountSelectedData > 1 ? 'Multivariate' : 'Univariate'}}</small>
          </div>
        </v-card-title>
        <v-card-text>
          <v-list>
            <v-list-tile @click="!data || addToProcessList(data, i)"
            v-for="(data, i) in selectHeaderDialog.data" :key="i">
              <v-list-tile-action v-if="i === mainKey">
                <v-icon color="yellow">star</v-icon>
              </v-list-tile-action>
              <v-list-tile-content>
                <v-list-tile-title>{{i}}</v-list-tile-title>
                <v-list-tile-sub-title class="blue--text text--lighten-2">{{data || 'it is not a valid data, a number is needed'}}</v-list-tile-sub-title>
              </v-list-tile-content>
              <v-list-tile-action v-if="selectHeaderDialog.selectedHeaders[i]">
                <v-icon color="green">check_circle</v-icon>
              </v-list-tile-action>
            </v-list-tile>
          </v-list>
        </v-card-text>
        <v-card-actions>
          <small><v-icon small color="yellow">star</v-icon> data to forecast</small>
          <v-spacer></v-spacer>
          <v-btn flat @click="reset">cancel</v-btn>
          <v-btn :disabled="amountSelectedData === 0" flat @click="dataFileToDataSet">process</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
export default {
  name: 'csvLoader',
  data: () => ({
    selectHeaderDialog: {
      selectedHeaders: {},
      value: false,
      data: {}
    },
    mainKey: null
  }),
  methods: {
    loadCSVFile (e) {
      const file = e.target.files[0]
      const reader = new FileReader()
      const result = {}
      reader.onload = (f) => {
        const rows = f.target.result.split('\n')
        const headers = rows[0].split(',')
        for (let h = 0; h < headers.length; h++) {
          result[headers[h]] = []
        }
        for (let v = 1; v < rows.length - 1; v++) {
          const values = rows[v].split(',')
          for (let i = 0; i < values.length; i++) {
            const val = +values[i]
            if (!isNaN(val)) {
              result[headers[i]].push(val)
            } else {
              result[headers[i]] = false
            }
          }
        }
        this.selectHeaderDialog.data = result
        this.selectHeaderDialog.value = true
      }
      reader.readAsBinaryString(file)
    },
    addToProcessList (val, key) {
      if (this.selectHeaderDialog.selectedHeaders[key]) {
        if (this.mainKey === key) this.mainKey = null
        this.$delete(this.selectHeaderDialog.selectedHeaders, key)
      } else {
        if (!this.mainKey) this.mainKey = key
        this.$set(this.selectHeaderDialog.selectedHeaders, key, val)
      }
    },
    dataFileToDataSet () {
      const selected = this.selectHeaderDialog.selectedHeaders
      // const amountOfColumns = Object.keys(selected).length
      const dataToProcess = []
      for (const key in selected) {
        dataToProcess.push({
          data: selected[key],
          name: key
        })
      }
      this.$emit('loaded', dataToProcess)
      this.reset()
    },
    reset () {
      this.mainKey = null
      this.selectHeaderDialog = {
        selectedHeaders: {},
        value: false,
        data: {}
      }
      this.$refs.csvFile.value = ''
    }
  },
  watch: {
    amountSelectedData: function (val) {
      if (val === 1) {
        this.$emit('serie', false)
      } else if (val > 1) {
        this.$emit('serie', true)
      }
    }
  },
  computed: {
    amountSelectedData () {
      return Object.keys(this.selectHeaderDialog.selectedHeaders).length
    }
  }
}
</script>