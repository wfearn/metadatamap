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
    selectedTopic: {},
    loading: false,
    maxTopic: 100.0,
    },
  mounted: function () {
    this.loading = true;

    this.getVocab();
    this.sendUpdate();
    // axios.get('/testDocs').then(response => {
    //   this.docs = response.data.docs;
    //   this.labels = reponse.data.labels;
    //   this.topics = reponse.data.topics;
    // }).catch(error => {
    //     console.log('error in /testDocs');
    // });

    this.loading = false;

    // var self = this;
    // $.ajax({
    //  // url: '/dist',
    //   url: '/testDocs',
    //   method: 'GET',
    //   success: function (data){
    //     console.log(data);
    //     self.docs = data.docs;
    //     self.labels = data.labels;
    //     self.topics = data.topics;

    //     var colorsList = ['#0000FF', '#8B0000', '#228B22', '#4B0082', '#FFA500', '#008080', '#FF4500'];
    //     var lenColors = colorsList.length;
    //     for (var i=0; i<self.labels.length; i++){
    //       Vue.set(self.colors, self.labels[i], colorsList[i%lenColors]);
    //     }
    //     self.maxTopic = self.findMaxTopic()
    //     self.loading = false;
    //   },
    //   error: function(error){
    //     console.log(error);
    //     self.error = true;
    //     self.loading = false;
    //     alert('AJAX Error');
    //   }
    // });
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
    // Data is expected to be sent in this form:
    // data = {anchor_tokens: [[token_str,..],...]
    //         labeled_docs: [{doc_id: number
    //                         user_label: label},...]
    //         unlabeled_docs: [doc_id,...]
    //        }
      axios.post('/api/update', {
        anchor_tokens: [],
        labeled_docs: this.labeledDocs.map(doc => {doc_id: doc.docId}),
                                         //          user_label: doc.userLabel}),
        unlabeled_docs: this.unlabeledDocs.map(doc => doc.docId),
      }).then(response => {
        console.log(response);
        this.data = response.data;
        this.anchors = response.data.anchors;
        this.unlabeledDocs = response.data.unlabeledDocs;
        this.labeledDocs = response.data.labeledDocs;
      }).catch(error => {
        console.log('Error in /api/update');
        console.log(error);
      });
    },
    colSize: function(label){
      var count = 0;
      for (var i=0; i<docs.length; i++){
        count += docs[i].label === label ? 1 : 0;
      }
      return count;
    },
    filterDocs: function(label){
      return this.docs.filter(function(doc){
        return doc.label === label;
      });
    },
    findMaxTopic: function(){
      console.log('MAXTOPIC');
      var max = 0;
      var self = this
      for (var i=0; i<this.topics.length; i++){
        var arr = this.docs.map(obj => obj[this.topics[i].topic]);
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
      this.unselectTopic(this.selectedTopic);
      if (this.selectedDoc.docNum === doc.docNum){
        this.unselectDocument(doc);
      }
      else{
        this.unselectDocument(this.selectedDoc)
        this.selectedDoc = doc;
        doc.selected = true;
        $('#doc'+doc.docNum).addClass('selected');
      }
    },
    unselectDocument(doc){
      $('#doc'+doc.docNum).removeClass('selected');
      this.selectedDoc = {};
      doc.selected = false;
    },
    selectTopic: function(topic){
      this.unselectDocument(this.selectedDoc);
      if (this.selectedTopic.topicNum === topic.topicNum){
      this.unselectTopic(topic);
      }
      else {
      this.selectedTopic = topic;
      }
    },
    unselectTopic: function(topic){
      this.selectedTopic = {};
    },
    getDocByNumber: function(number){
      for (var i=0; i<this.docs.length; i++){
        if (this.docs[i].docNum === number)
          return this.docs[i];
      }
    },
    heatmap: function(value, color){
      //if (!Array.isArray(rgb))
      //  var rgb = [rgb.r, rgb.g, rgb.b];
      //rgb=[52,119,220]
      if (typeof(color) === "string"){
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
      return this.heatmap(this.maxTopic/fraction, this.colors[label]);
    },
  }, //End methods
});
