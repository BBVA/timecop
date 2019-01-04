<template>
  <v-flex xs12>
    <v-card>
      <v-card-text>
        <v-flex xs12>
          <v-text-field label="Url" v-model="url" :rules="[rules.url]" outline></v-text-field>
        </v-flex>
        <v-flex xs12>
          <v-textarea hint="Paste your data or load CSV file" persistent-hint
          outline label="DatatSet" v-model="dataToProcess" :rules="[rules.json]"></v-textarea>
        </v-flex>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <csv-loader @loaded="processCSV" @serie="changeUrl"/>
      </v-card-actions>
    </v-card>
    <!-- parameters dialog -->
    <v-dialog v-model="parametersDialog.active"
      persistent
      width="550">
      <v-card>
        <v-card-title class="headline">Parameters</v-card-title>
        <v-card-text>
          <v-list three-line>
            <v-list-tile v-for="item in parametersList" :key="item.key">
              <v-list-tile-content>
                <v-list-tile-title>{{item.title}}</v-list-tile-title>
                <v-list-tile-sub-title>{{item.subtitle}}</v-list-tile-sub-title>
              </v-list-tile-content>
              <v-list-tile-action>
                <v-text-field
                  v-if="item.type === 'n' || item.type === 's'"
                  :style="{width: item.type === 'n' ? '48px' : '260px'}"
                  single-line persistent-hint full-width outline
                  v-model="item.value" />
                <v-switch
                  v-else
                  v-model="item.value"
                ></v-switch>
              </v-list-tile-action>
            </v-list-tile>
          </v-list>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn flat 
            @click="resetParametersDialog">cancel</v-btn>
          <v-btn flat color="green" @click="formatData">submit</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <!-- loading dialog -->
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
    <!-- error dialog -->
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
  </v-flex>
</template>

<script>
// import resultTest from '@/assets/dataTest/dataResponseUni'
// import resultTest from '@/assets/dataTest/dataResponseMulti'
import csvLoader from '@/components/csvLoader'
export default {
  name: 'Tform',
  components: {
    csvLoader
  },
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
    parametersDialog: {
      active: false,
      data: []
    },
    parametersList: [
      {
        title: 'Name',
        subtitle: 'Stores the list of points sent with that name and concatenates them to the existing ones before starting the prediction',
        value: '',
        type: 's',
        key: 'name'
      },
      {
        title: 'Future',
        subtitle: 'Steps in the future that you want to predict',
        value: 5,
        type: 'n',
        key: 'num_future'
      },
      {
        title: 'Deviation metric',
        subtitle: 'Anomaly sensitivity number',
        value: 2,
        type: 'n',
        key: 'desv_metric'
      },
      {
        title: 'Train',
        subtitle: '',
        value: true,
        type: 'boolean',
        key: 'train'
      },
      {
        title: 'Restart',
        subtitle: '',
        value: true,
        type: 'boolean',
        key: 'restart'
      }
    ],
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
    formatData () {
      const d = this.parametersDialog.data
      let dToSent = {}
      if (d.length > 0) {
        if (d.length === 1) {
          dToSent.data = d[0].data
        } else {
          dToSent.main = d[0].data
          dToSent.timeseries = []
          d.map((v, i) => {
            if (i > 1) dToSent.timeseries.push(v)
          })
        }
        this.parametersList.map(v => {
          if (v.value !== '') {
            dToSent[v.key] = v.value
          }
        })
        this.dataToProcess = JSON.stringify(dToSent)
        this.resetParametersDialog()
        this.getUrl()
      }
    },
    processCSV (e) {
      this.parametersDialog.active = true
      this.parametersDialog.data = e
    },
    getUrl () {
      // this.$emit('response', {dataToProcess: this.dataSet, result: resultTest})
      this.loading = true
      this.$http.post(this.url, this.dataSet).then(response => {
        this.$emit('response', {dataToProcess: this.dataSet, result: response.body})
        this.loading = false
      }).catch(err => {
        this.loading = false
        this.errorDialog.value = true
        this.errorDialog.text = err
        console.log(err)
      })
    },
    changeUrl (e) {
      if (e) {
        this.url = this.url.replace(/univariate/gi, 'multivariate')
      } else {
        this.url = this.url.replace(/multivariate/gi, 'univariate')
      }
    },
    resetParametersDialog () {
      this.parametersDialog.active = false
      this.parametersDialog.data = []
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
