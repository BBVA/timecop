webpackJsonp([1],{0:function(t,e){},"7zck":function(t,e){},NHnr:function(t,e,a){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var i=a("/5sW"),r={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("v-app",{attrs:{dark:t.dark}},[a("v-content",[a("router-view",{on:{toggleTheme:function(e){t.dark=e}}})],1)],1)},staticRenderFns:[]};var s=a("VU/8")({name:"App",data:function(){return{dark:!1}}},r,!1,function(t){a("etxH")},null,null).exports,n=a("/ocq"),o=a("fZjL"),l=a.n(o),d=a("mvHQ"),c=a.n(d),h={name:"csvLoader",data:function(){return{selectHeaderDialog:{selectedHeaders:{},value:!1,data:{}},mainKey:null}},methods:{loadCSVFile:function(t){var e=this,a=t.target.files[0],i=new FileReader,r={};i.onload=function(t){for(var a=t.target.result.split("\n"),i=a[0].split(","),s=0;s<i.length;s++)r[i[s]]=[];for(var n=1;n<a.length-1;n++)for(var o=a[n].split(","),l=0;l<o.length;l++){var d=+o[l];isNaN(d)?r[i[l]]=!1:r[i[l]]&&r[i[l]].push(d)}e.selectHeaderDialog.data=r,e.selectHeaderDialog.value=!0},i.readAsBinaryString(a)},addToProcessList:function(t,e){this.selectHeaderDialog.selectedHeaders[e]?(this.mainKey===e&&(this.mainKey=null),this.$delete(this.selectHeaderDialog.selectedHeaders,e)):(this.mainKey||(this.mainKey=e),this.$set(this.selectHeaderDialog.selectedHeaders,e,t))},dataFileToDataSet:function(){var t=this.selectHeaderDialog.selectedHeaders,e=[];for(var a in t)e.push({data:t[a],name:a});this.$emit("loaded",e),this.reset()},reset:function(){this.mainKey=null,this.selectHeaderDialog={selectedHeaders:{},value:!1,data:{}},this.$refs.csvFile.value=""}},watch:{amountSelectedData:function(t){1===t?this.$emit("serie",!1):t>1&&this.$emit("serie",!0)}},computed:{amountSelectedData:function(){return l()(this.selectHeaderDialog.selectedHeaders).length}}},u={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",[a("v-btn",{attrs:{outline:"",color:"blue"},on:{click:function(e){t.$refs.csvFile.click()}}},[t._v("load csv")]),t._v(" "),a("input",{ref:"csvFile",attrs:{type:"file",hidden:"",accept:".csv, text/plain"},on:{change:t.loadCSVFile}}),t._v(" "),a("v-dialog",{attrs:{persistent:"","max-width":"500"},model:{value:t.selectHeaderDialog.value,callback:function(e){t.$set(t.selectHeaderDialog,"value",e)},expression:"selectHeaderDialog.value"}},[a("v-card",{attrs:{dark:""}},[a("v-card-title",[a("div",[a("h3",{staticClass:"headline"},[t._v("What column do you want to process?")]),t._v(" "),t.amountSelectedData>0?a("small",[t._v("data type: "+t._s(t.amountSelectedData>1?"Multivariate":"Univariate"))]):t._e()])]),t._v(" "),a("v-card-text",[a("v-list",t._l(t.selectHeaderDialog.data,function(e,i){return a("v-list-tile",{key:i,on:{click:function(a){!e||t.addToProcessList(e,i)}}},[i===t.mainKey?a("v-list-tile-action",[a("v-icon",{attrs:{color:"yellow"}},[t._v("star")])],1):t._e(),t._v(" "),a("v-list-tile-content",[a("v-list-tile-title",[t._v(t._s(i))]),t._v(" "),a("v-list-tile-sub-title",{staticClass:"blue--text text--lighten-2"},[t._v(t._s(e||"not valid data"))])],1),t._v(" "),t.selectHeaderDialog.selectedHeaders[i]?a("v-list-tile-action",[a("v-icon",{attrs:{color:"green"}},[t._v("check_circle")])],1):t._e()],1)}))],1),t._v(" "),a("v-card-actions",[a("small",[a("v-icon",{attrs:{small:"",color:"yellow"}},[t._v("star")]),t._v(" data to forecast")],1),t._v(" "),a("v-spacer"),t._v(" "),a("v-btn",{attrs:{flat:""},on:{click:t.reset}},[t._v("cancel")]),t._v(" "),a("v-btn",{attrs:{disabled:0===t.amountSelectedData,flat:""},on:{click:t.dataFileToDataSet}},[t._v("process")])],1)],1)],1)],1)},staticRenderFns:[]},v={name:"Tform",components:{csvLoader:a("VU/8")(h,u,!1,null,null,null).exports},data:function(){return{url:"http://localhost:5000/univariate/get",dataToProcess:"",loading:!1,future:5,rules:{json:function(t){try{JSON.parse(t)}catch(t){return"Data is not a valid json"}return!0},url:function(t){return!!/(http|https):\/\/(\w+:{0,1}\w*)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%!\\/]))?/.test(t)||"Url is not a valid"}},errorDialog:{value:!1,text:""},parametersDialog:{active:!1,data:[]},parametersList:[{title:"Name",subtitle:"Stores the list of points sent with that name and concatenates them to the existing ones before starting the prediction",value:"",type:"s",key:"name"},{title:"Future",subtitle:"Steps in the future that you want to predict",value:5,type:"n",key:"num_future"},{title:"Deviation metric",subtitle:"Anomaly sensitivity number",value:2,type:"n",key:"desv_metric"}],selectHeaderDialog:{selectedHeaders:{},value:!1,data:{}},multivariateData:{timeseries:[],main:[]},mainKey:null}},methods:{formatData:function(){var t=this.parametersDialog.data,e={};t.length>0&&(1===t.length?e.data=t[0].data:(e.main=t[0].data,e.timeseries=[],t.map(function(t,a){a>1&&e.timeseries.push(t)})),this.parametersList.map(function(t){t.value&&(e[t.key]=t.value)}),this.dataToProcess=c()(e),this.resetParametersDialog(),this.getUrl())},processCSV:function(t){this.parametersDialog.active=!0,this.parametersDialog.data=t},getUrl:function(){var t=this;this.loading=!0,this.$http.post(this.url,this.dataSet).then(function(e){t.$emit("response",{dataToProcess:t.dataSet,result:e.body}),t.loading=!1}).catch(function(e){t.loading=!1,t.errorDialog.value=!0,t.errorDialog.text=e,console.log(e)})},changeUrl:function(t){this.url=t?this.url.replace(/univariate/gi,"multivariate"):this.url.replace(/multivariate/gi,"univariate")},resetParametersDialog:function(){this.parametersDialog.active=!1,this.parametersDialog.data=[]}},computed:{dataSet:function(){return JSON.parse(this.dataToProcess)},amountSelectedData:function(){return l()(this.selectHeaderDialog.selectedHeaders).length}}},g={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("v-flex",{attrs:{xs12:""}},[a("v-card",[a("v-card-text",[a("v-flex",{attrs:{xs12:""}},[a("v-text-field",{attrs:{label:"Url",rules:[t.rules.url],outline:""},model:{value:t.url,callback:function(e){t.url=e},expression:"url"}})],1),t._v(" "),a("v-flex",{attrs:{xs12:""}},[a("v-textarea",{attrs:{hint:"Paste your data or load CSV file","persistent-hint":"",outline:"",label:"DatatSet",rules:[t.rules.json]},model:{value:t.dataToProcess,callback:function(e){t.dataToProcess=e},expression:"dataToProcess"}})],1)],1),t._v(" "),a("v-card-actions",[a("v-spacer"),t._v(" "),a("csv-loader",{on:{loaded:t.processCSV,serie:t.changeUrl}})],1)],1),t._v(" "),a("v-dialog",{attrs:{persistent:"",width:"550"},model:{value:t.parametersDialog.active,callback:function(e){t.$set(t.parametersDialog,"active",e)},expression:"parametersDialog.active"}},[a("v-card",[a("v-card-title",{staticClass:"headline"},[t._v("Parameters")]),t._v(" "),a("v-card-text",[a("v-list",{attrs:{"three-line":""}},t._l(t.parametersList,function(e){return a("v-list-tile",{key:e.key},[a("v-list-tile-content",[a("v-list-tile-title",[t._v(t._s(e.title))]),t._v(" "),a("v-list-tile-sub-title",[t._v(t._s(e.subtitle))])],1),t._v(" "),a("v-list-tile-action",[a("v-text-field",{style:{width:"n"===e.type?"48px":"260px"},attrs:{"single-line":"","persistent-hint":"","full-width":"",outline:""},model:{value:e.value,callback:function(a){t.$set(e,"value",a)},expression:"item.value"}})],1)],1)}))],1),t._v(" "),a("v-card-actions",[a("v-spacer"),t._v(" "),a("v-btn",{attrs:{flat:""},on:{click:t.resetParametersDialog}},[t._v("cancel")]),t._v(" "),a("v-btn",{attrs:{flat:"",color:"green"},on:{click:t.formatData}},[t._v("submit")])],1)],1)],1),t._v(" "),a("v-dialog",{attrs:{"hide-overlay":"",persistent:"",width:"300"},model:{value:t.loading,callback:function(e){t.loading=e},expression:"loading"}},[a("v-card",{attrs:{"max-width":"300"}},[a("v-card-text",[t._v("\n        Processing... this may take a while\n        "),a("v-progress-linear",{staticClass:"mb-0",attrs:{indeterminate:""}})],1)],1)],1),t._v(" "),a("v-dialog",{attrs:{"hide-overlay":"",persistent:"",width:"500"},model:{value:t.errorDialog.value,callback:function(e){t.$set(t.errorDialog,"value",e)},expression:"errorDialog.value"}},[a("v-card",{attrs:{color:"red",dark:"","max-width":"500"}},[a("v-card-text",[a("pre",[t._v(t._s(t.errorDialog.text))])]),t._v(" "),a("v-card-actions",[a("v-spacer"),t._v(" "),a("v-btn",{attrs:{flat:""},on:{click:function(e){t.errorDialog.value=!1}}},[t._v("ok")])],1)],1)],1)],1)},staticRenderFns:[]},m=a("VU/8")(v,g,!1,null,null,null).exports,p=a("pFYg"),f=a.n(p),x={name:"jsonViewer",props:["json"],data:function(){return{items:[],hightLevel:0}},methods:{addItems:function(t,e,a){for(var i in t)this.items.splice(e,0,{name:i,type:f()(t[i]),data:t[i],level:a,open:!0});this.hightLevel=a},toggle:function(t,e,a,i){i?this.addItems(t,e,a):this.deleteItems(t,e,a)},deleteItems:function(t,e,a){this.hightLeve--;for(var i=this.items.filter(function(t){return t.level>=a}).length,r=l()(t).length,s=a!==this.hightLevel?i:r,n=0;n<s;n++)this.items.splice(e,1)},getColor:function(t){switch(t){case"object":return"green--text darken-4";case"string":return"red--text darken-4";case"number":return"blue--text darken-4";case"boolean":return"indigo--text darken-4"}}},watch:{json:{handler:function(t){this.items=[],this.addItems(t,0,0),this.items.reverse()},immediate:!0}}},y={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("v-flex",{attrs:{xs12:""}},t._l(t.items,function(e,i){return a("div",{key:i,style:{"margin-left":10*e.level+"px",cursor:"object"===e.type&&0!==e.data.length?"pointer":"auto"},on:{click:function(a){"object"!==e.type||t.toggle(e.data,i+1,e.level+1,e.open),e.open=!e.open}}},[a("span",{class:{"font-weight-bold":"object"===e.type&&0!==e.data.length}},[t._v(t._s(e.name)+":")]),t._v(" "),a("span",{class:t.getColor(e.type)},[t._v(t._s("object"===e.type&&0!==e.data.length?"{...}":e.data))])])}))},staticRenderFns:[]},_=a("VU/8")(x,y,!1,null,null,null).exports,k={props:{dataSet:null,height:{type:Number,default:500},marginLeft:{type:Number,default:40},marginTop:{type:Number,default:40},marginRight:{type:Number,default:20},marginBottom:{type:Number,default:30},toggleSize:{type:Boolean},background:String},data:function(){return{width:null,chartWidth:1e3,chartHeight:400,zoomMin:0,zoomMax:100,panEnabled:!1,offsetX:0,markPos:{pos:0,val:0},toGraph:{},total:[],extendedArea:{active:!1,ready:!1,value:"",el:{x:0,y:0,w:500,h:230,draggable:!1,ctrlRigth:!1,ctrlBottom:!1}}}},mounted:function(){this.calculateSize()},methods:{calculateSize:function(){this.width=this.$el.clientWidth,this.chartWidth=this.width-(this.marginLeft+this.marginRight),this.chartHeight=this.height-(this.marginTop+this.marginBottom)},drawData:function(t){var e=t.toPredict.main||t.toPredict.data;this.$set(this.toGraph,"main",{data:e.map(function(t,e){return{x:e,y:+t}}),visible:!0,color:this.$utils.getRandomColor(),name:"main"});var a=t.toPredict.timeseries;if(a&&a.length)for(var i=0;i<a.length;i++)this.$set(this.toGraph,"data-"+i,{data:a[i].data.map(function(t,e){return{x:e,y:+t}}),visible:!0,color:this.$utils.getRandomColor(),name:"data-"+i});if(t.prediction.debug)for(var r in t.prediction.debug){var s=t.prediction.debug[r];this.$set(this.toGraph,r,{data:s.map(function(t){return{x:+t.step,y:+t["expected value"]||+t.valores||+t.var_0||+t.values||+t.value||+t.Prediction}}),visible:!0,color:this.$utils.getRandomColor(),name:r,debug:!0})}var n=t.prediction.future;this.$set(this.toGraph,"prediction",{data:n.map(function(t,e){return{x:+t.step,y:+t["expected value"]||+t.value||+t.valores||+t.var_0||+t.values}}),visible:!0,color:this.$utils.getRandomColor(),name:"prediction"}),this.toGraph.prediction.data.unshift({x:this.toGraph.main.data[this.toGraph.main.data.length-1].x,y:this.toGraph.main.data[this.toGraph.main.data.length-1].y}),this.zoomMax=e.length+n.length,this.total=this.toGraph.main.data.concat(this.toGraph.prediction.data)},zoom:function(t){if(!this.extendedArea.active){var e=t.wheelDelta?.02*t.wheelDelta:-t.deltaY;this.zoomMin+=e,this.zoomMax+=-e}},pan:function(t){if(this.panEnabled&&!this.extendedArea.active)this.offsetX+=-t.movementX,this.markPos=0;else if(this.total.length>0)for(var e=t.offsetX-this.marginLeft,a=0;a<this.total.length;a++){var i=this.$utils.scale(a,this.zoomMin,this.zoomMax,this.chartWidth)-this.offsetX;i<e&&(this.markPos={pos:i,val:a})}},moveExtendedArea:function(t){this.extendedArea.el.draggable&&(this.extendedArea.el.x+=t.movementX,this.extendedArea.el.y+=t.movementY)},randomizeColors:function(){var t=this;for(var e in this.toGraph)this.toGraph[e].color=this.$utils.getRandomColor();this.$nextTick(function(){t.extendedArea.value=t.$refs["graph-container"].innerHTML})},reset:function(){this.zoomMin=0,this.zoomMax=100,this.offsetX=0,this.toGraph={},this.extendedArea.active=!1}},watch:{dataSet:{handler:function(t){var e=this;t.toPredict&&t.prediction&&this.$nextTick(function(){e.reset(),e.drawData(t),e.$nextTick(function(){e.extendedArea.value=e.$refs["graph-container"].innerHTML,e.extendedArea.ready=!0})})},immediate:!0},toggleSize:function(){this.calculateSize()},extendedAreaState:function(t){t&&(this.extendedArea.value=this.$refs["graph-container"].innerHTML)}},computed:{anomalies:function(){var t=this.dataSet;if(t.toPredict){for(var e=[],a=0;a<t.prediction.past.length;a++){if(t.prediction.past[a].step){var i=this.$utils.scale(t.prediction.past[a].step,this.zoomMin,this.zoomMax,this.chartWidth);e.push(i)}}return e}},globalMax:function(){var t=-1e11;for(var e in this.toGraph)if(this.toGraph[e].visible){var a=this.$utils.getMax(this.toGraph[e].data,"y");t=t>a?t:a}return t},globalMin:function(){var t=1e11;for(var e in this.toGraph)if(this.toGraph[e].visible){var a=this.$utils.getMin(this.toGraph[e].data,"y");t=t<a?t:a}return t},extendedAreaState:function(){return this.extendedArea.active}}},b={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("v-flex",{directives:[{name:"resize",rawName:"v-resize",value:t.calculateSize,expression:"calculateSize"}],attrs:{xs12:""}},[a("v-toolbar",{attrs:{dense:""}},[a("v-toolbar-items",t._l(t.toGraph,function(e,i){return a("v-btn",{key:"btn"+i,class:{"graph-inactive-btn":!e.visible},attrs:{color:e.color,disabled:t.extendedArea.active,flat:""},on:{click:function(t){e.visible=!e.visible}}},[t._v("\n        "+t._s(e.name)+"\n        "),e.name===t.dataSet.prediction.engine?a("span",{staticClass:"winner"},[t._v("♕")]):t._e()])})),t._v(" "),a("v-spacer"),t._v(" "),a("v-tooltip",{attrs:{bottom:""}},[a("v-btn",{attrs:{slot:"activator",disabled:t.extendedArea.active,flat:"",icon:""},on:{click:t.randomizeColors},slot:"activator"},[a("v-icon",[t._v("brush")])],1),t._v(" "),a("span",[t._v("Randomize Colors")])],1),t._v(" "),a("v-tooltip",{attrs:{bottom:""}},[a("v-btn",{attrs:{slot:"activator",disabled:0===t.total.length,flat:"",icon:""},on:{click:function(e){t.extendedArea.active=!t.extendedArea.active}},slot:"activator"},[a("v-icon",[t._v("crop_free")])],1),t._v(" "),a("span",[t._v("extend area")])],1)],1),t._v(" "),a("svg",{ref:"graph-container",class:t.background,attrs:{width:"100%",height:t.height},on:{wheel:function(e){return e.preventDefault(),t.zoom(e)},mousedown:function(e){if(e.ctrlKey||e.shiftKey||e.altKey||e.metaKey)return null;t.panEnabled=!0},mouseup:function(e){t.panEnabled=!1},mousemove:t.pan}},[t.total.length>0?a("g",{attrs:{transform:"translate("+t.marginLeft+", "+t.marginTop+")"}},[a("defs",[a("clipPath",{attrs:{id:"clip-rect"}},[a("rect",{attrs:{width:t.chartWidth,height:t.height,y:-this.marginTop}})])]),t._v(" "),a("g",{attrs:{"clip-path":"url(#clip-rect)"}},[t._l(t.toGraph,function(e,i){return e.visible?a("g",{key:i},[a("c-path",{attrs:{dasharray:e.debug?"5,5":"",transform:"translate("+-t.offsetX+", 0)",color:e.color,rangeX:[t.zoomMin,t.zoomMax],rangeY:[t.globalMin,t.globalMax],dataset:e.data,y:"y",x:"x",height:t.chartHeight,width:t.chartWidth}})],1):t._e()}),t._v(" "),t.markPos.pos>0?a("line",{attrs:{x1:t.markPos.pos,x2:t.markPos.pos,y2:t.chartHeight,"stroke-width":"2",stroke:"#0eff0e78",fill:"none"}}):t._e(),t._v(" "),t.markPos.pos>0?a("text",{attrs:{fill:"#0eff0e78","text-anchor":"middle",dy:"-5px",transform:"translate("+t.markPos.pos+" 0)"}},[t._v("\n        "+t._s(t.markPos.val)+"\n      ")]):t._e(),t._v(" "),t._l(t.anomalies,function(e,i){return a("circle",{key:i,attrs:{cx:e-t.offsetX,cy:t.chartHeight,r:"7",stroke:"white","stroke-width":"1",fill:"red"}})}),t._v(" "),a("c-axis-x",{attrs:{transform:"translate("+-t.offsetX+" "+t.chartHeight+")",range:[t.zoomMin,t.zoomMax],dataset:t.total,x:"x",ticks:25,fixed:1,height:t.chartHeight,width:t.chartWidth,strokeColor:this.$vuetify.dark?"white":"#6d6d6d"}})],2),t._v(" "),a("c-axis-y",{attrs:{transform:"translate("+(t.chartWidth-t.marginLeft-t.marginRight)+" 0)",range:[t.globalMin,t.globalMax],ticks:5,fixed:3,height:t.chartHeight,strokeColor:this.$vuetify.dark?"white":"#6d6d6d"}})],1):t._e(),t._v(" "),t.extendedArea.active?a("g",{attrs:{transform:"translate("+t.extendedArea.el.x+", "+t.extendedArea.el.y+")"}},[a("rect",{attrs:{fill:"#ffffff17",width:t.extendedArea.el.w,height:t.extendedArea.el.h},on:{mousemove:t.moveExtendedArea,mousedown:function(e){if(e.ctrlKey||e.shiftKey||e.altKey||e.metaKey)return null;t.extendedArea.el.draggable=!0},mouseup:function(e){t.extendedArea.el.draggable=!1},mouseout:function(e){t.extendedArea.el.draggable=!1}}}),t._v(" "),a("circle",{attrs:{cx:t.extendedArea.el.w,cy:t.extendedArea.el.h/2,r:"15",stroke:"black",fill:"grey"},on:{mousemove:function(e){t.extendedArea.el.w+=e.movementX}}}),t._v(" "),a("circle",{attrs:{cx:t.extendedArea.el.w/2,cy:t.extendedArea.el.h,r:"15",stroke:"black",fill:"grey"},on:{mousemove:function(e){t.extendedArea.el.h+=e.movementY}}})]):t._e()]),t._v(" "),a("svg",{class:t.background,attrs:{viewBox:t.extendedArea.el.x+" "+t.extendedArea.el.y+" "+t.extendedArea.el.w+" "+t.extendedArea.el.h,width:"100%",height:"300",preserveAspectRatio:"xMidYMid slice"},domProps:{innerHTML:t._s(t.extendedArea.value)}})],1)},staticRenderFns:[]};var w={name:"homeView",components:{tForm:m,tJson:_,tGraph2d:a("VU/8")(k,b,!1,function(t){a("P6lf")},null,null).exports},data:function(){return{response:{},toggleDataVisibility:!0,dark:!0}},mounted:function(){this.$emit("toggleTheme",this.dark)},methods:{toggleData:function(){this.toggleDataVisibility=!0},showResponse:function(t){this.response={toPredict:t.dataToProcess,prediction:t.result}}}},D={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("v-container",{attrs:{fluid:"","grid-list-md":""}},[a("v-toolbar",{attrs:{dense:"",app:""}},[a("span",{staticClass:"mt-3"},[a("v-switch",{on:{change:function(e){t.$emit("toggleTheme",t.dark)}},model:{value:t.dark,callback:function(e){t.dark=e},expression:"dark"}})],1),t._v(" "),t.dark?a("img",{staticClass:"pa-1",attrs:{src:"static/img/logo_dark.svg",height:"70%",alt:"Time Cop"}}):a("img",{staticClass:"pa-1",attrs:{src:"static/img/logo.svg",height:"70%",alt:"Time Cop"}}),t._v(" "),a("v-spacer"),t._v(" "),a("v-btn",{attrs:{flat:"",color:"blue"},on:{click:function(e){t.toggleDataVisibility=!t.toggleDataVisibility}}},[t._v("\n      data\n      "),t.toggleDataVisibility?a("v-icon",{attrs:{right:""}},[t._v("visibility")]):a("v-icon",{attrs:{right:""}},[t._v("visibility_off")])],1)],1),t._v(" "),a("v-layout",{attrs:{wrap:""}},[a("v-flex",{class:t.toggleDataVisibility?"xs8":"xs12"},[a("t-graph-2d",{attrs:{dataSet:t.response,toggleSize:t.toggleDataVisibility,height:350,"margin-left":5,background:t.dark?"grey darken-3":"grey lighten-3"}})],1),t._v(" "),a("v-flex",{directives:[{name:"show",rawName:"v-show",value:t.toggleDataVisibility,expression:"toggleDataVisibility"}],attrs:{xs4:""}},[a("t-form",{staticClass:"mb-4",on:{response:t.showResponse}}),t._v(" "),a("t-json",{attrs:{json:t.response.prediction}})],1)],1)],1)},staticRenderFns:[]};var $=a("VU/8")(w,D,!1,function(t){a("R5Fh")},null,null).exports;i.default.use(n.a);var S=new n.a({routes:[{path:"/",name:"home",component:$}]}),A=a("3EgV"),H=a.n(A),M=a("8+8L"),P=a("wmFm");a("7zck");i.default.use(H.a),i.default.config.productionTip=!1,i.default.use(M.a),i.default.http.headers.common["content-type"]="application/json",i.default.use(P.a),new i.default({el:"#app",router:S,render:function(t){return t(s)}})},NvMd:function(t,e,a){"use strict";var i={name:"cAxisY",props:{range:{type:Array,required:!0},ticks:{type:Number},height:{type:Number,required:!0},label:{type:String},fixed:{type:Number},strokeColor:{type:String,default:"white"}},computed:{ticksList:function(){if(this.range&&this.range.length>0){var t=[],e=this.range[0],a=(this.range[1]-e)/(this.ticks-1),i=e;t.push(this.fixed?i.toFixed(this.fixed):i);for(var r=1;r<this.ticks;r++)t.push(this.fixed?(i+=a).toFixed(this.fixed):i+=a);return t.reverse()}}}},r={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("g",{attrs:{"text-anchor":"end",fill:t.strokeColor,stroke:t.strokeColor,"stroke-width":"1"}},[a("line",{attrs:{y1:t.height}}),t._v(" "),t._l(t.ticksList,function(e,i){return a("g",{key:"tick"+i,attrs:{transform:"translate(0, "+i*(t.height/(t.ticksList.length-1))+")"}},[a("line",{attrs:{x2:"-6"}}),t._v(" "),a("text",{attrs:{"stroke-width":"0.1",x:"-9",dy:"0.32em"}},[t._v(t._s(e))])])}),t._v(" "),a("text",{attrs:{transform:"rotate(-90)",y:"6",dy:"0.71em","stroke-width":"0.1"}},[t._v(t._s(t.label))])],2)},staticRenderFns:[]},s=a("VU/8")(i,r,!1,null,null,null);e.a=s.exports},OBgj:function(t,e,a){"use strict";var i={name:"cPath",props:{dataset:{type:Array,required:!0},y:{type:String,required:!0},x:{type:String,required:!0},height:{type:Number,required:!0},width:{type:Number,required:!0},color:{type:String,default:"steelblue"},strokeWidth:{type:String,default:"1.5"},dasharray:{type:String,default:""},rangeY:{type:Array},rangeX:{type:Array}},computed:{points:function(){if(this.dataset&&this.dataset.length>0){for(var t=this.rangeY?this.rangeY[0]:this.$utils.getMin(this.dataset,this.y),e=this.rangeY?this.rangeY[1]:this.$utils.getMax(this.dataset,this.y),a=this.rangeX?this.rangeX[0]:this.$utils.getMin(this.dataset,this.x),i=this.rangeX?this.rangeX[1]:this.$utils.getMax(this.dataset,this.x),r=[],s=[],n=0;n<this.dataset.length;n++)r.push(this.$utils.scale(this.dataset[n][this.x],a,i,this.width)),s.push(this.$utils.scale(this.dataset[n][this.y],t,e,this.height,!0));r.reverse(),s.reverse();for(var o="M"+r[0]+" "+s[0],l=1;l<this.dataset.length;l++)o+=" L"+r[l]+" "+s[l]+" ";return o}}}},r={render:function(){var t=this.$createElement;return(this._self._c||t)("path",{attrs:{fill:"none",stroke:this.color,"stroke-linejoin":"round","stroke-linecap":"round","stroke-width":this.strokeWidth,"stroke-dasharray":this.dasharray,d:this.points}})},staticRenderFns:[]},s=a("VU/8")(i,r,!1,null,null,null);e.a=s.exports},P6lf:function(t,e){},R5Fh:function(t,e){},etxH:function(t,e){},meHP:function(t,e,a){"use strict";var i={name:"cAxisX",props:{dataset:{type:Array,required:!0},range:{type:Array},x:{type:String},ticks:{type:Number},height:{type:Number,required:!0},width:{type:Number,required:!0},label:{type:String},strokeColor:{type:String,default:"white"}},computed:{ticksList:function(){if(this.dataset&&this.dataset.length>0){for(var t=this.range?this.range[0]:this.$utils.getMin(this.dataset,this.x),e=this.range?this.range[1]:this.$utils.getMax(this.dataset,this.x),a=[],i=0;i<this.dataset.length;i++)a.push({position:this.$utils.scale(this.dataset[i][this.x],t,e,this.width)});for(var r=0;r<this.dataset.length;r+=Math.round(this.dataset.length/this.ticks))a[r].value=this.dataset[r][this.x];return a.reverse()}}}},r={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("g",{attrs:{"text-anchor":"middle",fill:t.strokeColor,stroke:t.strokeColor,"stroke-width":"1",transform:"translate(0, "+t.height+")"}},[t.ticksList&&t.ticksList.length>0?a("line",{attrs:{x1:t.ticksList[0].position,x2:t.ticksList[t.ticksList.length-1].position}}):t._e(),t._v(" "),t._l(t.ticksList,function(e,i){return a("g",{key:"tick"+i,attrs:{transform:"translate("+e.position+", 0)"}},[a("line",{attrs:{y2:"6"}}),t._v(" "),a("text",{attrs:{"stroke-width":"0.1",y:"9",dy:"0.71em"}},[t._v(t._s(e.value))])])}),t._v(" "),a("text",{attrs:{x:t.ticksList[0].position,y:"-25",dx:"-0.71em",dy:"0.71em","stroke-width":"0.1"}},[t._v(t._s(t.label))])],2)},staticRenderFns:[]},s=a("VU/8")(i,r,!1,null,null,null);e.a=s.exports},o3Cr:function(t,e,a){"use strict";var i={name:"bars",props:{dataset:{type:Array,required:!0},column:{type:String,required:!0},height:{type:Number,required:!0},width:{type:Number,required:!0}},computed:{list:function(){for(var t=[],e=this.$utils.getMax(this.dataset,this.column),a=this.$utils.getMin(this.dataset,this.column),i=0;i<this.dataset.length;i++)t.push({d:this.dataset[i][this.column],v:this.$utils.scale(this.dataset[i][this.column],a,e,this.height,!0)});return t}}},r={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("g",t._l(t.list,function(e,i){return a("rect",{key:i,attrs:{x:i*(t.width/t.list.length),y:e.v,width:30,height:t.height-e.v,fill:"green"}})}))},staticRenderFns:[]},s=a("VU/8")(i,r,!1,null,null,null);e.a=s.exports},"rw8+":function(t,e,a){"use strict";var i={name:"cCircle",props:{x:{type:[Number,String],required:!0},y:{type:[Number,String],required:!0},r:{type:[Number,String],required:!0},strokeColor:{type:String,default:"black"},strokeWidth:{type:[Number,String],default:3},color:{type:String,default:"red"}}},r={render:function(){var t=this.$createElement;return(this._self._c||t)("circle",{attrs:{cx:this.x,cy:this.y,r:this.r,stroke:this.strokeColor,"stroke-width":this.strokeWidth,fill:this.color}})},staticRenderFns:[]},s=a("VU/8")(i,r,!1,null,null,null);e.a=s.exports}},["NHnr"]);
//# sourceMappingURL=app.01ce87f6520592c64594.js.map