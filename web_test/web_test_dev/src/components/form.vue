<template>
  <v-flex xs12>
    <v-card>
      <v-card-text>
        <v-flex xs12>
          <v-text-field label="Url" v-model="url" :rules="[rules.url]" outline @keypress.enter="getUrl"></v-text-field>
        </v-flex>
        <v-flex xs12>
          <v-textarea hint="Paste your data or load CSV file" persistent-hint
          outline label="DatatSet" v-model="dataToProcess" :rules="[rules.json]" @keypress.enter="getUrl"></v-textarea>
        </v-flex>
        <v-layout>
          <v-flex xs4>
            <v-text-field
            outline single-line persistent-hint
            v-model="future"></v-text-field>
          </v-flex>
          <v-flex xs10 class="mt-3">Steps in the future that you want to predict</v-flex>
        </v-layout>
      </v-card-text>
      <v-card-actions>
        <v-btn outline color="indigo" @click="$refs.csvFile.click()">csv</v-btn>
        <input type="file" hidden ref="csvFile" accept=".csv, text/plain" @change="loadCSVFile">
        <v-spacer></v-spacer>
        <v-btn @click="getUrl" color="success">Submit <v-icon right>send</v-icon></v-btn>
      </v-card-actions>
    </v-card>
    <v-dialog
      v-model="loading"
      hide-overlay
      persistent
      width="300"
    >
      <v-card max-width="300">
        <v-card-text>
          Processing... this may take a while
          <v-progress-linear
            indeterminate
            class="mb-0"
          ></v-progress-linear>
        </v-card-text>
      </v-card>
    </v-dialog>
    <v-dialog
      v-model="errorDialog.value"
      hide-overlay
      persistent
      width="500"
    >
      <v-card color="red" dark max-width="500">
        <v-card-text>
          <pre>{{errorDialog.text}}</pre>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn flat @click="errorDialog.value = false">ok</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-dialog
      v-model="selectHeaderDialog.value"
      persistent
      width="500"
    >
      <v-card dark max-width="500">
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
          <v-btn flat @click="selectHeaderDialog.value = false">cancel</v-btn>
          <v-btn :disabled="amountSelectedData === 0" flat @click="dataFileToDataSet">process</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-flex>
</template>

<script>
import resultTest from '@/assets/dataTest/dataResponseUni'
// import resultTest from '@/assets/dataTest/dataResponseMulti'
export default {
  name: 'Tform',
  data: () => ({
    url: 'http://localhost:5000/univariate/get',
    dataToProcess: '',
    loading: false,
    future: 5,
    rules: {
      json: v => {
        try {
          JSON.parse(v)
        } catch (e) {
          return 'Data is not a valid json'
        }
        return true
      },
      url: v => {
        var regex = /(http|https):\/\/(\w+:{0,1}\w*)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%!\\/]))?/
        if (!regex.test(v)) {
          return 'Url is not a valid'
        } else {
          return true
        }
      }
    },
    errorDialog: {
      value: false,
      text: ''
    },
    selectHeaderDialog: {
      selectedHeaders: {},
      value: false,
      data: {}
    },
    multivariateData: {
      timeseries: [],
      main: []
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
    dataFileToDataSet () {
      const selected = this.selectHeaderDialog.selectedHeaders
      const amountOfColumns = Object.keys(selected).length
      if (amountOfColumns === 1) {
        for (const key in selected) {
          this.dataToProcess = `{"data": [${selected[key]}]}`
        }
      } else {
        for (const key in selected) {
          if (this.multivariateData.main.length > 0) {
            this.multivariateData.timeseries.push({data: selected[key]})
          } else {
            this.multivariateData.main = selected[key]
          }
        }
        this.dataToProcess = JSON.stringify(this.multivariateData)
      }
      // reset values
      this.multivariateData = {
        timeseries: [],
        main: []
      }
      this.mainKey = ''
      this.selectHeaderDialog = {
        selectedHeaders: {},
        value: false,
        data: {}
      }
      this.$refs.csvFile.value = ''
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
    getUrl () {
      // this.$emit('response', {dataToProcess: this.dataSet, result: resultTest})
      this.loading = true
      this.$set(this.dataSet, 'num_future', +this.future)
      this.$http.post(this.url, this.dataSet).then(response => {
        this.$emit('response', {dataToProcess: this.dataSet, result: response.body})
        this.loading = false
      }).catch(err => {
        this.loading = false
        this.errorDialog.value = true
        this.errorDialog.text = err
        console.log(err)
      })
    }
  },
  computed: {
    dataSet () {
      return JSON.parse(this.dataToProcess)
    },
    amountSelectedData () {
      return Object.keys(this.selectHeaderDialog.selectedHeaders).length
    }
  }
}
</script>
