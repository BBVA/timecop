<template>
  <v-container fluid grid-list-md>
    <v-toolbar dense app>
      <span class="mt-3">
        <v-switch
          v-model="dark"
          @change="$emit('toggleTheme', dark)"
        ></v-switch>
      </span>
      <img v-if="!dark" src="static/img/logo.svg" height="70%" class="pa-1" alt="Time Cop">
      <img v-else src="static/img/logo_dark.svg" height="70%" class="pa-1" alt="Time Cop">
      <v-spacer></v-spacer>
      <v-btn light target="new" href="https://github.com/BBVA/timecop">
        <img class="mr-2" src="static/github.svg" height="26px" alt="github">
        <span>github</span>
      </v-btn>
      <v-btn flat color="blue" @click="toggleDataVisibility = !toggleDataVisibility">
        data
        <v-icon v-if="toggleDataVisibility" right>visibility</v-icon>
        <v-icon v-else right>visibility_off</v-icon>
      </v-btn>
    </v-toolbar>
    <v-layout wrap>
      <v-flex :class="toggleDataVisibility ? 'xs8' : 'xs12'">
        <t-graph-2d
        :triggerReset="reset"
        :dataSet="response"
        :toggleSize="toggleDataVisibility"
        :height="350"
        :margin-left="5"
        :background="dark ? 'grey darken-3': 'grey lighten-3'"/>
      </v-flex>
      <v-flex xs4 v-show="toggleDataVisibility">
        <t-form @response="showResponse" @reset="reset = !reset" class="mb-4"></t-form>
        <t-json :json="response.prediction"></t-json>
      </v-flex>
    </v-layout>
  </v-container>
</template>

<script>
import tForm from '@/components/form'
import tJson from '@/components/jsonViewer'
import tGraph2d from '@/components/graph'
export default {
  name: 'homeView',
  components: {
    tForm,
    tJson,
    tGraph2d
  },
  data: () => ({
    response: {},
    toggleDataVisibility: true,
    dark: true,
    reset: false
  }),
  mounted () {
    this.$emit('toggleTheme', this.dark)
  },
  methods: {
    toggleData () {
      this.toggleDataVisibility = true
    },
    showResponse (e) {
      this.response = {
        toPredict: e.dataToProcess,
        prediction: e.result
      }
    }
  }
}
</script>

<style>
.column-code {
  height: 85vh;
  overflow-y: scroll;
}
</style>
