import Vue from 'vue'
import App from './App'
import router from './router'
import Vuetify from 'vuetify'
import VueResource from 'vue-resource'
// import chartConstructor from './constructor/2d_graph'
import chart2dConstructor from 'vue-chart2d-constructor'
import 'vuetify/dist/vuetify.min.css'

Vue.use(Vuetify)

Vue.config.productionTip = false
Vue.use(VueResource)
Vue.http.headers.common['content-type'] = 'application/json'
Vue.use(chart2dConstructor)
/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  render: h => h(App)
})
