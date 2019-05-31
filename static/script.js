
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
    showModal: true,
    autocompleteInput: '',
    autocompleteResults: [],
    userId: '',
    inputId: '',
    showAnswers: false,
    showAnchorInfo: false,
    canEditAnchors: false,
    showTokens: false,
    displayInstructions: false,
    // TODO: currently randomly choosing these conditions, but need to ensure that we get equal numbers in all conditions, so instead should use server to track how many participants of each condition
    inputUncertainty: Math.random() >= 0.5,
    perceivedControl: Math.random() >= 0.5,
    labeledCount: 0,
    logText: '',
    startDate: null,
    timer: null,
    totalTime: 20*60*1000,
    time: 0,
    firstPage: true, // track which page of modal the user is viewing
    started: false, // track whether the user has started the task
    finished: false, // track whether user has finished the task
    clickedSurvey: false, // track whether the user has clicked the survey link
    finishedSurvey: false // track whether the user has proceeded to the task after completing the survey
    },
  components: {
  //  'modal': Modal,
  },
  mounted: function () {
    // console.log('mounted')

    // We dont need vocab for this study
    //this.loading = true;
    //this.getVocab();
    //this.sendUpdate();

    // is this the on load function?
    console.log('mounted');
    this.getNewUser();

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
    prettyTime: function(){
      let seconds = parseInt(this.time / 1000);
      let remaining = ('0' + (seconds % 60)).slice(-2);
      return parseInt(seconds/60) + ':' + remaining;
    },
  }, //End computed
  watch: {
  }, //End watch
  methods: {
    getVocab: function(){
      axios.get('/api/vocab').then(response => {
        this.vocab = response.data.vocab;
      }).catch(error => {
        console.error('error in /api/vocab', error)
      });
    },
    getIdData: function(id){
      if (id === ''){
        alert('That user id was not found');
        return;
      }
      axios.get('/api/checkuserid/'+id).then(response => {
        this.checked = response.data.hasId;
        if (response.data.hasId){
          this.userId = id;
          this.sendUpdate();
        }
        else{
          alert('That user id was not found');
        }
      }).catch(error => {
        console.error('error in /api/checkuserid', error)
      });
    },
    getNewUser: function(){
      // console.log('getNewUser');
      axios.post('/api/adduser').then(response => {
        this.userId = response.data.userId;
      //  this.sendUpdate();
      }).catch(error => {
        console.error('error in /api/adduser', error);
        // console.log(error);
      });
    },
    sendUpdate: function(){
      if (this.finished){
        return;
      }
      console.log('sendUpdate');
      this.logText += this.getExactTime() + '||SEND_UPDATE||';
    // Data is expected to be sent to server in this form:
    // data = {anchor_tokens: [[token_str,..],...]
    //         labeled_docs: [{doc_id: number
    //                         user_label: label},...]
    //        }
      this.loading = true;
      var curLabeledDocs = this.unlabeledDocs.filter(
                  doc => doc.hasOwnProperty('userLabel'))
                         .map(doc => ({doc_id: doc.docId,
                                       user_label: doc.userLabel.slice(0, -1)}));
      // Something like this?
      for (var i=0; i<this.unlabeledDocs.length; i++){
        let d = this.unlabeledDocs[i];
        this.logText += ('(' + d.docId + ',' + (d.hasOwnProperty('userLabel') ? d.userLabel : 'Unlabeled') +
                         (i<this.unlabeledDocs.length-1 ? ') ' : ')'));
      }
      this.logText += '\n';
      // Or maybe like this?
      // this.logText += 'LABELEDDOCS||';
      // for (var i=0; i<curLabeledDocs.length; i++){
      //   this.logText += ('(' + curLabeledDocs[i].doc_id + ',' + curLabeledDocs[i].user_label +
      //                    (i<curLabeledDocs.length-1 ? '), ' : ')'));
      // }
      // this.logText += '\n' + '-'.repeat(10) + '\n\n';

      this.labeledCount += curLabeledDocs.length
      axios.post('/api/update', {
        anchor_tokens: this.anchors.map(anchorObj => (anchorObj.anchorWords)),
        //labeled_docs: this.labeledDocs.map(doc => ({doc_id: doc.docId,
        //                                            user_label: doc.userLabel})),
        labeled_docs: curLabeledDocs,
        user_id: this.userId,
        // updates the log text on call to update
        log_text: this.logText,
      }).then(response => {
        console.log(response);
        this.updateData = response.data;
        this.anchors = response.data.anchors;
        // new set of unlabeled documents
        this.unlabeledDocs = response.data.unlabeledDocs;
        // AMR 5/24: shuffle the order randomly (needed for teaming study)
        this.shuffle(this.unlabledDocs);
        this.labels = response.data.labels;
        this.labeled_docs = [];
        this.loading = false;
        this.isMounted = true;
        if (!this.colorsChosen){
          this.chooseColors();
          this.colorsChosen = true;
        }
        for (var i=0; i<this.unlabeledDocs.length; i++){
          Vue.set(this.unlabeledDocs[i], 'open', false);
        }
        // TODO: check the current system accuracy
        // this.getAccuracy();
      }).catch(error => {
        console.log('Error in /api/update');
        console.log(error);
      });
    },//end sendUpdate function
    getAccuracy: function(){
      console.log('getAccuracy');
      this.loading = true;
      axios.post('/api/accuracy', {
        anchor_tokens: this.anchors.map(anchorObj => (anchorObj.anchorWords))
      }).then(response => {
        console.log('current accuracy', response.data.accuracy)
        this.accuracy = response.data.accuracy;
        this.loading = false;
      }).catch(error => {
        console.error('Error in /api/accuracy', error);
        this.loading = false;
      });
    },
    shuffle: function(array) {
      if (!array) {
        return;
      }
      var currentIndex = array.length;
      var temporaryValue, randomIndex;

      // While there remain elements to shuffle...
      while (0 !== currentIndex) {
        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
      }

      return array;
    },
    chooseColors: function(){
      console.log('chooseColors');
      //var colorsList = ['#191919', '#FE8000','#191919', , '#FE8000','#8B0000', '#4C4CFF','#0000FF', '#228B22', '#4B0082', '#FFA500', '#008080', '#FF4500'];
      //Christmas
      //Halloween
      //colorsList = ['#191919', '#FE8000']

      // var lenColors = colorsList.length;
      // for (var i=0; i<this.labels.length; i++){
      //   console.log(this.labels[i].label, colorsList[i%lenColors]);
      //   Vue.set(this.colors, this.labels[i].label, colorsList[i%lenColors]);
      // }
      if (this.labels[0].label === 'negative' ||
          this.labels[0].label === 'positive'){
        var colorsList = ['#bb2528', '#146b3a'];
        Vue.set(this.colors, 'negative', colorsList[0]);
        console.log(this.colors)
        Vue.set(this.colors, 'positive', colorsList[1]);
        console.log(this.colors)
        console.log(colorsList)
      } else {
        // original
        //var colorsList = ['#0015bc', '#e9141d'];

        // lighter shade
        //var colorsList = ['#6673D6', '#F17278'];

        // lighterer shade
        //var colorsList = ['#848FDE', '#F38E93'];

        // lightererer shade
        var colorsList = ['#A8A8FD', '#F38E93'];

        Vue.set(this.colors, 'D', colorsList[0]);
        console.log(this.colors)
        Vue.set(this.colors, 'R', colorsList[1]);
        console.log(this.colors)
        console.log(colorsList)
      }
    },
    colSize: function(label){
      var count = 0;
      for (var i=0; i<docs.length; i++){
        count += docs[i].label === label ? 1 : 0;
      }
      return count;
    },
    closeModal: function(){
      if (this.started){
        this.showModal=false;
        this.logText += (this.getExactTime() + '||CLOSE_INSTRUCTIONS \n');
      }
    },
    openModal: function(){
      this.showModal=true;
      this.firstPage=true;
    },
    toggleModal: function(){
      if(this.showModal){
        this.closeModal()
      } else {
        this.logText += (this.getExactTime() + '||OPEN_INSTRUCTIONS \n');
        this.openModal()
      }
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
      if (!((item.hasOwnProperty('docId') &&
             this.drag.hasOwnProperty('docId'))
            ||
            (item.hasOwnProperty('anchorId') &&
             this.drag.hasOwnProperty('anchorId'))
         )){ return; }

      console.log('dropItem');
      var indexItem = arr.indexOf(this.drag);
      var indexTarget = arr.indexOf(item);
      arr.splice(indexItem,1);
      arr.splice(indexTarget,0,this.drag);

      if (this.drag.hasOwnProperty('docId')){
        this.unselectAllDocs();
        this.selectDocument(this.drag);
      }


      $('#' + id).removeClass('dragover');
      Vue.set(this.drag, 'dragging', false);
    },
    dragOver: function(item, id, arr){
      if (!((item.hasOwnProperty('docId') &&
             this.drag.hasOwnProperty('docId'))
            ||
            (item.hasOwnProperty('anchorId') &&
             this.drag.hasOwnProperty('anchorId'))
         )){ return; }
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
        this.unselectAllDocs();
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
    unselectAllDocs: function(){
      for (var i=0; i<this.unlabeledDocs.length; i++){
        this.unselectDocument(this.unlabeledDocs[i]);
      }
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
    getLabelCount: function(labelId){
      var label = this.labels[labelId].label
      return this.labels[labelId].count + this.labeledDocs.filter(doc => doc.userLabel === label).length;
    },
    assignDocLabel: function(label){
      if (!(this.drag.hasOwnProperty('docId'))){
        return;
      }

      this.unselectAllDocs();

      var doc = this.drag;
      Vue.set(doc, 'userLabel', label.label);
      this.labeledDocs.push(doc);
      var index = this.unlabeledDocs.indexOf(doc);
      this.unlabeledDocs.splice(index, 1)

      for (var key in label.anchorIdToValue){
        if (label.anchorIdToValue.hasOwnProperty(key)){
          let newVal = (label.anchorIdToValue[key]*label.count + doc.anchorIdToValue[key])/(label.count+1);
          Vue.set(label.anchorIdToValue, key, newVal);
        }
      }
      Vue.set(label, 'count', label.count+1);

    },
    heatmap: function(value, color, max=this.maxTopic){
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
      pos = Math.round((value/max)*nGroups).toFixed(0);

      getColor = function(cIndex){
        return ((rgbEnd[cIndex] + ((pos * (rgb[cIndex] - rgbEnd[cIndex])) / (nGroups-1))).toFixed(0));
      };
      var clr = 'rgb('+getColor(0)+','+getColor(1)+','+getColor(2)+')';
      return clr;
    },
    lightenDarkenColor: function(color, val){
      if (typeof(color) === 'string'){
        var rgb = hexToRgb(color);
        rgb = [rgb.r, rgb.g, rgb.b];
      }
      for (var i=0; i<3; i++){
        rgb[i] -= val;
        rgb[i] = Math.min(255, Math.max(0, rgb[i]));
      }
      return '#' + (rgb[2] | rgb[1]<<8 | rgb[0]<<16).toString(16);
    },
    labelColor(label, fraction=3){
      return 'blue';
      //return this.heatmap(this.maxTopic/fraction, this.colors[label]);
    },
    onAutocompleteChange: function(input){
      if (input.length<4){
        return;
      }
      this.autocompleteResults = this.vocab.filter(word => {
        return word.toLowerCase().indexOf(input.toLowerCase()) > -1});
    },
    addWord: function(e, anchor){
      var word = anchor.autocompleteInput.toLowerCase().trim();
      console.log('attempt to add', word);
      if (this.vocab.includes(word)){
        anchor.anchorWords.push(word);
        anchor.autocompleteInput = "";
      }
      else{
        alert(word + ' is not in the vocabulary');
      }
    },
    deleteWord: function(anchor, index){
      console.log(index);
      anchor.anchorWords.splice(index, 1);
    },
    addAnchor: function(){
      var anchor = {anchorId: 'x',
                    anchorWords: [],
                    topicWords: []};
      this.anchors.push(anchor);
    },
    deleteAnchor: function(anchorIndex){
      this.anchors.splice(anchorIndex, 1);
    },
    pad: function(num, size){
      return ('000000' + num).substr(-size);
    },
    expandLabel: function(label){
      if (label === 'D') return 'Dem';
      if (label === 'R') return 'Rep';
    },
    labelAllCorrect: function(){
      for(var i=0; i<this.unlabeledDocs.length; i++){
        Vue.set(this.unlabeledDocs[i], 'userLabel', this.unlabeledDocs[i].trueLabel+'1');
      }
    },
    getDocHtml: function(doc){
      console.log('Getting HTML');
      var html = '';
      var prev = 0
      var loc;
      var label;
      var a;
      var b;
      for (var i=0; i<doc.highlights.length; i++){
        loc = doc.highlights[i][0];
        label = doc.highlights[i][1];
        a = loc[0];
        b = loc[1];
        html += doc.text.substr(prev, a-prev);
        html += ('<span class="rounded" style="background-color: ' + this.colors[label] +
                      '">' + doc.text.substr(a, b-a) + '</span> ');
        prev = b;
      }
      html += doc.text.substr(prev, doc.text.length);
      return html;
    },
    deleteLabel: function(doc){
      Vue.delete(doc, 'userLabel');
    },
    labelDoc: function(doc, label){
      if (doc.hasOwnProperty('userLabel')){
        if (doc.userLabel === label){
          this.deleteLabel(doc);
          this.logText += (this.getExactTime() + '||UNLABEL_DOC||' + doc.docId +  '\n');
          return;
        }
      }
      Vue.set(doc, 'userLabel', label);
      this.logText += (this.getExactTime() + '||LABEL_DOC||' + doc.docId + ',' + label + '\n');
    },
    getConfidenceWord: function(doc){
      return doc.prediction.confidence < .9 ? 'Maybe' : 'Definitely';
    },
    toggleDocOpen: function(doc){
      if(doc.open){
        this.logText += (this.getExactTime() + '||CLOSE_DOC||' + doc.docId +  '\n');
      } else {
        this.logText += (this.getExactTime() + '||OPEN_DOC||' + doc.docId +  '\n');
      }
      doc.open = !doc.open;
    },
    getExactTime: function(){
      return new Date() - this.startDate;
    },
    startTask: function(){
      // this.getNewUser();
      this.sendUpdate();
      this.finished = false;
      this.time = this.totalTime;
      this.twoMinute = setTimeout( () => {
        this.logText += (this.getExactTime() + '||TIME_WARNING \n');
        // TODO: show in modal
        alert('You have 2 minutes remaining to confirm or correct the system predictions. At the end of task time, all outstanding {{this.perceivedControl ? "assignments" : "suggestions"}} will be saved to the system.');
      }, this.totalTime - 2*60*1000);
      this.timer = setInterval( () => {
        if (this.time > 0) {
          this.time -= 1000;
        } else {
          clearInterval(this.timer);
          this.logText += (this.getExactTime() + '||TIME_UP \n');
          // send final update
          this.sendUpdate();
          // set finished status to true
          this.finished = true;
          // open the modal
          this.openModal();
        }
      }, 1000);
      this.showModal = false;
      this.started = true;
      this.logText += (this.getExactTime() + '||STARTING_TASK||' + this.userId + '||c,' + this.perceivedControl + ',u,' + this.inputUncertainty + '\n');
      // Event listener to close the modal on Esc
      document.addEventListener("keydown", (e) => {
        if (this.showModal && e.keyCode == 27) {
          this.closeModal()
        }
      });

      //this.maxTopic = this.findMaxTopic()
      this.startDate = new Date();

    }

  }, //End methods
});
