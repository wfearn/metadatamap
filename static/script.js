
function hexToRgb(hex) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

var app = new Vue({
  el: '#app',
  data: {
    unlabeledDocs: [],
    labeledDocs: [],
    labels: [],
    anchors: [],
    vocab: [], // Corpus vocabulary (for determining which words can be anchors)
    colors: {}, // TODO figure out what to do with this
    drag: {},
    selectedDoc: {},
    selectedAnchor: {},
    loading: false,
    maxTopic: 1.000,
    isMounted: false,
    colorsChosen: false,
    showModal: false,
    },
  components: {
  //  'modal': Modal,
  },
  mounted: function () {
    this.loading = true;
    this.getVocab();
    this.sendUpdate();

    // Event listener to close the modal on Esc
    document.addEventListener("keydown", (e) => {
      if (this.showModal && e.keyCode == 27) {
        this.closeModal()
      }
    });

    console.log('HELLO')
    //this.maxTopic = this.findMaxTopic()

  }, //End mounted
  computed: {
    docsByLabel: function(){
      docsByLabelObj = {}
      console.log(this.labels);
      for(var i=0; i<this.labels.length; i++){
        Vue.set(docsByLabelObj, this.labels[i], this.filterDocs(this.labels[i]));
      }
      return docsByLabelObj;
    },
  }, //End computed
  methods: {
    getVocab: function(){
      axios.get('/api/vocab').then(response => {
        this.vocab = response.data.vocab;
      }).catch(error => {
        console.log('error in /api/vocab')
      });
    },
    sendUpdate: function(){
    // Data is expected to be sent to server in this form:
    // data = {anchor_tokens: [[token_str,..],...]
    //         labeled_docs: [{doc_id: number
    //                         user_label: label},...]
    //        }
      this.loading = true;
      axios.post('/api/update', {
        anchor_tokens: this.anchors.map(anchorObj => (anchorObj.anchorWords)),
        labeled_docs: this.labeledDocs.map(doc => ({doc_id: doc.docId,
                                                    user_label: doc.userLabel})),
      }).then(response => {
        console.log(response);
        this.updateData = response.data;
        this.anchors = response.data.anchors;
        this.unlabeledDocs = response.data.unlabeledDocs;
        this.labels = response.data.labels;
        this.loading = false;
        this.isMounted = true;
        if (!this.colorsChosen){
          this.chooseColors();
          this.colorsChosen = true;
        }

      }).catch(error => {
        console.log('Error in /api/update');
        console.log(error);
      });
    },//end sendUpdate function
    chooseColors: function(){
      var colorsList = ['#191919', '#FE8000','#191919', , '#FE8000','#8B0000', '#4C4CFF','#0000FF', '#228B22', '#4B0082',
                        '#FFA500', '#008080', '#FF4500'];
      //Christmas
      colorsList = ['#bb2528', '#146b3a']
      //Halloween
      //colorsList = ['#191919', '#FE8000']

      // var lenColors = colorsList.length;
      // for (var i=0; i<this.labels.length; i++){
      //   console.log(this.labels[i].label, colorsList[i%lenColors]);
      //   Vue.set(this.colors, this.labels[i].label, colorsList[i%lenColors]);
      // }
      Vue.set(this.colors, 'positive', colorsList[0]);
      Vue.set(this.colors, 'negative', colorsList[1]);
    },
    colSize: function(label){
      var count = 0;
      for (var i=0; i<docs.length; i++){
        count += docs[i].label === label ? 1 : 0;
      }
      return count;
    },
    closeModal: function(){
      this.showModal=false;
    },
    openModal: function(){
      this.showModal=true;
    },
    filterDocs: function(label){
      return this.docs.filter(function(doc){
        return doc.label === label;
      });
    },
    findMaxTopic: function(){
      console.log('MAXTOPIC');
      var max = 0;
      for (var i=0; i<this.anchors.length; i++){
        var arr = this.docs.map(obj => obj[this.anchors[i].topic]);
        var newMax = Math.max.apply(Math, arr);
        max = (newMax>max) ? newMax : max;
      }
      console.log('EndMAX');
      return max;
    },
    //timeout
    //https://schier.co/blog/2014/12/08/wait-for-user-to-stop-typing-using-javascript.html
    dragItem: function(item, id, arr){
      console.log('dragItem');
      this.drag = item;
      Vue.set(item,'dragging', true);
      $('#'+id).addClass('dragging');
    },
   // dropItem: function(item, arr){
   //   var indexItem = this.docs.indexOf(this.drag);
   //   var indexTarget = this.docs.indexOf(item);
   //   this.docs.splice(indexItem,1);
   //   this.docs.splice(indexTarget,0,this.drag);
   //   el = $('#'+item.docNum);
   //   el.removeClass('dragover');
   // },
    dropItem: function(item, id, arr){
      console.log('dropItem');
      var indexItem = arr.indexOf(this.drag);
      var indexTarget = arr.indexOf(item);
      arr.splice(indexItem,1);
      arr.splice(indexTarget,0,this.drag);
      $('#' + id).removeClass('dragover');
      Vue.set(this.drag, 'dragging', false);
    },
    dragOver: function(item, id, arr){
      console.log('dragOver');
      $('#'+id).addClass('dragover');
      console.log(id);
      var indexItem = arr.indexOf(this.drag);
      var indexTarget = arr.indexOf(item);
      arr.splice(indexItem,1);
      arr.splice(indexTarget,0,this.drag);
    },
    dragIntoDivider: function(label){
      console.log('------------');
      console.log(label);
      Vue.set(this.drag, 'label', label);
    },
    dragLeave: function(item, id, arr){
      console.log('dragLeave');
      $('#'+id).removeClass('dragover');
    },
    dragEnd: function(item, id, arr){
      console.log('dragEnd');
      console.log(id);
      $('#'+id).removeClass('dragging');
      Vue.set(this.drag, 'dragging', false);
    },
    selectDocument: function(doc){
      this.unselectAnchor(this.selectedAnchor);
      if (this.selectedDoc.docId === doc.docId){
        this.unselectDocument(doc);
      }
      else{
        this.unselectDocument(this.selectedDoc)
        this.selectedDoc = doc;
        doc.selected = true;
        $('#doc'+doc.docId).addClass('selected');
      }
    },
    unselectDocument(doc){
      $('#doc'+doc.docId).removeClass('selected');
      this.selectedDoc = {};
      doc.selected = false;
    },
    selectAnchor: function(anchor){
      this.unselectDocument(this.selectedDoc);
      if (this.selectedAnchor.anchorId === anchor.anchorId){
      this.unselectAnchor(anchor);
      }
      else {
      this.selectedAnchor = anchor;
      }
    },
    unselectAnchor: function(anchor){
      this.selectedAnchor = {};
    },
    getDocById: function(number){
      for (var i=0; i<this.unlabeledDocs.length; i++){
        if (this.unlabeledDocs[i].docId === number)
          return this.unlabeledDocs[i];
      }
    },
    sortAnchors: function(label){
      this.anchors.sort((a,b) => {
        return label.anchorIdToValue[a.anchorId] - label.anchorIdToValue[b.anchorId]
      });
    },
    heatmap: function(value, color){
      //if (!Array.isArray(rgb))
      //  var rgb = [rgb.r, rgb.g, rgb.b];
      //rgb=[52,119,220]
      //return 'blue'
      //return   '#FE8000'
      //return '#191919'

      if (typeof(color) === 'string'){
        var rgb = hexToRgb(color);
        rgb = [rgb.r, rgb.g, rgb.b];
      }

      var nGroups = 100;
      var rgbEnd = [255, 255, 255];
      var max = this.maxTopic;
      pos = Math.round((value/max)*nGroups).toFixed(0);

      getColor = function(cIndex){
        return ((rgbEnd[cIndex] + ((pos * (rgb[cIndex] - rgbEnd[cIndex])) / (nGroups-1))).toFixed(0));
      };
      var clr = 'rgb('+getColor(0)+','+getColor(1)+','+getColor(2)+')';
      return clr;
    },
    labelColor(label, fraction=3){
      return 'blue';
      //return this.heatmap(this.maxTopic/fraction, this.colors[label]);
    },
  }, //End methods
});

