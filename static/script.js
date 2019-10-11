
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
    numCorrect: 0, // track the total number of correctly predicted documents the user was exposed to
    totalDocs: 0, // track the total number of predicted documents the user was exposed to
    // TODO: currently randomly choosing these conditions, but need to ensure that we get equal numbers in all conditions, so instead should use server to track how many participants of each condition
    // if perceived control is true, that means it's the assign condition; if input uncertainty is true, that means it's the four option condition
    inputUncertainty: Math.random() >= 0.5,
    perceivedControl: Math.random() >= 0.5,
    labeledCount: 0,
    correctDocumentDelta: 0,
    logText: '',
    startDate: null,
    timer: null,
    totalTime: 20*60*1000, // total time is 20 minutes
    time: 0, // initially, time is 0
    paused: false, // track when the user is on the instructions or alert page (at which time we pause the task)
    timeWarning: false, // track whether the user should see the time warning alert
    modalState: 0,
    secondPage: false,
    started: false, // track whether the user has started the task
    finished: false, // track whether user has finished the task
    refreshed: false, // track whether the system has just updated with new debates
    inputProvided: false, // track whether the user provided input on the last round
    clickedSurvey: false, // track whether the user has clicked the survey link
    finishedSurvey: false // track whether the user has proceeded to the task after completing the survey
  },
  components: {},
  mounted: function () {

  }, //End mounted
  computed: {
    docsByLabel: function() {
      docsByLabelObj = {}
      for(var i = 0; i < this.labels.length; i++){
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
  watch: {}, //End watch
  methods: {
    // determine which slider image to put in instructions given the perceived control condition
    getSliderUrl: function() {
      if (this.perceivedControl) {
        return '/static/images/spectrum-assign.png';
      } else {
        return '/static/images/spectrum-suggest.png';
      }
    },
    // determine which tool screenshot to provide given the condition
    getScreenshotUrl: function() {
      if (this.perceivedControl) {
        if (this.inputUncertainty) {
          // assign and four options
          return '/static/images/assign-four-screenshot.png'
        } else {
          // assign and two options
          return '/static/images/assign-two-screenshot.png'
        }
      } else {
        if (this.inputUncertainty) {
          // suggest and four options
          return '/static/images/suggest-four-screenshot.png'
        } else {
          // suggest and two options
          return '/static/images/suggest-two-screenshot.png'
        }
      }
    },
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
      axios.post('/api/adduser').then(response => {
        this.userId = response.data.userId;
          this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||INITIAL_LOAD||' + this.userId + '||c,' + this.perceivedControl + ',u,' + this.inputUncertainty + '\n');
          // include the below to hide the tutorial
         this.sendUpdate();
      }).catch(error => {
        console.error('error in /api/adduser', error);
        // console.log(error);
      });
    },
    sendUpdate: function(){
      if (this.finished){
        return;
      }
      this.logText += this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||SEND_UPDATE||labeled,';
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
      this.labeledCount += curLabeledDocs.length;
      this.logText += curLabeledDocs.length + ',total,' + this.labeledCount + '||';

      if (curLabeledDocs.length > 0) {
        this.inputProvided = true;

      } else {
        this.inputProvided = false;

      }
      // Something like this?
      var correctLabels = 0;
      var incorrectLabels = 0;

      for (var i = 0; i < this.unlabeledDocs.length; i++) {
        let d = this.unlabeledDocs[i];
        // score the user labels
        if (d.userLabel) {
          if (d.trueLabel === (d.userLabel.substring(0, d.userLabel.length - 1))) {
          correctLabels += 1;
          } else {
            incorrectLabels += 1;
          }
        }

        // determine how many R and D highlighted words
        let numD = 0;
        let numR = 0;
        for (var j = 0; j < d.highlights.length; j++) {
          let h = d.highlights[j];

          if (h[1] === 'D') {
            numD += 1;

          } else {
            numR += 1;

          }
        }

        // doc id, true label, system label, system label confidence, user label, highlights
        this.logText += ('doc,' + d.docId + ',true,' + d.trueLabel + ',pred,' + d.prediction.label + ',conf,' + d.prediction.confidence + ',user,' + (d.hasOwnProperty('userLabel') ? d.userLabel : 'Unlabeled') + ',highlights,' + d.highlights.length + ',D,' + numD + ',R,' + numR + ';');
      }
      // number of correct labels, number of incorrect labels (for the user)
      this.logText += "||correct," + correctLabels + ',incorrect,' + incorrectLabels;
      this.logText += '\n';

      axios.post('/api/update', {
        labeled_docs: curLabeledDocs,
        user_id: this.userId,

        // updates the log text on call to update
        log_text: this.logText,
      }).then(response => {
        this.updateData = response.data;
        this.anchors = response.data.anchors;

        // new set of unlabeled documents
        this.unlabeledDocs = response.data.unlabeledDocs;

        // determine the classifier accuracy for the returned set of documents, and track classifier accuracy for all documents the user has been exposed to

        this.correctDocumentDelta = response.data.correctDocumentDelta

        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||NEW_DEBATES||' + this.userId + '||currentAccuracy,' + this.correctDocumentDelta + '\n');

        // AMR 5/24: shuffle the order randomly (needed for teaming study)
        this.unlabeledDocs = this.shuffle(this.unlabeledDocs);
        this.labels = response.data.labels;
        this.labeled_docs = [];
        this.loading = false;
        this.isMounted = true;

        if (!this.colorsChosen){
          this.chooseColors();
          this.colorsChosen = true;
        }

        for (var i = 0; i < this.unlabeledDocs.length; i++) {
          Vue.set(this.unlabeledDocs[i], 'open', false);
        }

        // pop up the modal
        this.refreshed = true;
        this.openModal();
        // TODO: check the current system accuracy
        //this.getAccuracy();
      }).catch(error => {
        console.error('Error in /api/update', error);
      });

    },//end sendUpdate function
    getAccuracy: function(){
      this.loading = true;

      axios.post('/api/accuracy', {
          user_id: this.userId
      }).then(response => {
        if(response.data.accuracy) {
          console.log('current accuracy',  response.data.accuracy)
          this.accuracy = response.data.accuracy;
        }

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
    //  console.log('shuffled array', array);
      return array;
    },
    chooseColors: function(){

      if (this.labels[0].label === 'negative' || this.labels[0].label === 'positive') {
        var colorsList = ['#bb2528', '#146b3a'];
        Vue.set(this.colors, 'negative', colorsList[0]);

        Vue.set(this.colors, 'positive', colorsList[1]);

      } else {
        var colorsList = ['#A8A8FD', '#F38E93'];

        Vue.set(this.colors, 'D', colorsList[0]);
        Vue.set(this.colors, 'R', colorsList[1]);

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

    //  console.log('closing the modal!');
      if (this.started) {
        this.timeWarning = false;
        this.paused = false;
        this.showModal = false;
        this.refreshed = false;
        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||CLOSE_INSTRUCTIONS \n');
      }
    },
    openModal: function() {
      this.paused = true;
      this.showModal = true;
      this.modalState = 0;
    },
    toggleModal: function(){
      if(this.showModal){
        this.closeModal()
      } else {
        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||OPEN_INSTRUCTIONS \n');
        this.openModal()
      }
    },
    filterDocs: function(label){
      return this.docs.filter(function(doc){
        return doc.label === label;
      });
    },
    findMaxTopic: function(){
    //  console.log('MAXTOPIC');
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
    //  console.log('dragItem');
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

    //  console.log('dropItem');
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
    //  console.log('dragOver');
      $('#'+id).addClass('dragover');
    //  console.log(id);
      var indexItem = arr.indexOf(this.drag);
      var indexTarget = arr.indexOf(item);
      arr.splice(indexItem,1);
      arr.splice(indexTarget,0,this.drag);
    },
    dragIntoDivider: function(label){
    //  console.log('------------');
    //  console.log(label);
      Vue.set(this.drag, 'label', label);
    },
    dragLeave: function(item, id, arr){
    //  console.log('dragLeave');
      $('#'+id).removeClass('dragover');
    },
    dragEnd: function(item, id, arr){
    //  console.log('dragEnd');
    //  console.log(id);
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
  //    console.log('attempt to add', word);
      if (this.vocab.includes(word)){
        anchor.anchorWords.push(word);
        anchor.autocompleteInput = "";
      }
      else{
        alert(word + ' is not in the vocabulary');
      }
    },
    deleteWord: function(anchor, index){
  //    console.log(index);
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
      for(var i = 0; i < this.unlabeledDocs.length; i++){
        Vue.set(this.unlabeledDocs[i], 'userLabel', this.unlabeledDocs[i].trueLabel+'1');
      }
    },
    convertToRegex: function(ngrams) {
        anything = '[^a-zA-Z]+'
        //anything = '.*?'

        var regexBeginning = '(^|\\b)(';
        var fullRegex = regexBeginning;

        for(i = 0; i < ngrams.length; i++) {
            fullRegex += ngrams[i];
            if(i < (ngrams.length - 1)) {
                fullRegex += `${anything}`;
            }
        }

        fullRegex += ')($|\\b)';

        return fullRegex;
    },
    getDocHtml: function(doc){
  //    console.log('Getting HTML');
      var html = doc.text;
      var prev = 0
      var loc;
      var label;
      var a;
      var b;
      for (var i = 0; i < doc.highlights.length; i++){
          var ngram = doc.highlights[i][0];
          var label = doc.highlights[i][1];
          var ngrams_regex = this.convertToRegex(ngram.split(' '));
          var doc_label = this.colors[label];

          var re = new RegExp(ngrams_regex, 'g');
          html = html.replace(re, '$1<span class="rounded" style="background-color: ' + doc_label + '">$2</span>$3');

      }

      return html;
    },
    deleteLabel: function(doc){
      Vue.delete(doc, 'userLabel');
    },
    labelDoc: function(doc, label){
      if (doc.hasOwnProperty('userLabel')){
        if (doc.userLabel === label){
          this.deleteLabel(doc);
          this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||UNLABEL_DOC||' + doc.docId +  '\n');
          return;
        }
      }
      Vue.set(doc, 'userLabel', label);
      // timestamp, active time, label doc event, doc id, true label, system provided label, confidence, user provided label
      this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||LABEL_DOC||' + doc.docId + ',' + doc.trueLabel + ',' + doc.prediction.label + ',' + doc.prediction.confidence + ',' + label + '\n');
    },

    getConfidenceWord: function(doc){
      // TODO: need a better way to set this threshold..
      return doc.prediction.confidence < .95 ? 'Possibly' : 'Probably';
    },

    getConfidenceColor: function(doc) {
      if (doc.prediction.confidence < .95) {
        return this.lightenDarkenColor(this.colors[doc.prediction.label], -40);
      } else {
        return this.colors[doc.prediction.label];
      }
    },

    toggleDocOpen: function(doc){
      if(doc.open){
        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime()+ '||CLOSE_DOC||' + doc.docId +  '\n');
      } else {
        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||OPEN_DOC||' + doc.docId +  '\n');
      }
      doc.open = !doc.open;
    },

    getExactTime: function() {
  //    console.log('curr time', new Date())
      return new Date() - this.startDate;
    },
    getActiveTime: function() {
      return this.totalTime - this.time;
    },
    getCurrTimeStamp: function () {
      return new Date();
    },
    startTask: function() {
      console.log('starting the task!');
      this.startDate = new Date();

      // include the below to hide the tutorial
       this.getNewUser();
      // INITIAL UPDATE
      // comment out the below to hide the tutorial
    //  this.sendUpdate();
      this.finished = false;
      this.paused = false;
      this.time = this.totalTime;
      this.twoMinute = setTimeout( () => {
        this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||TIME_WARNING \n');
        // show in modal and pause task time
        this.timeWarning = true;
        this.openModal();
      }, this.totalTime - 2*60*1000);
      this.timer = setInterval( () => {
        if (this.time > 0) {
          if (!this.paused) {
            this.time -= 1000;
          }
        } else {
          clearInterval(this.timer);
          this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||TIME_UP \n');
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
      this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||STARTING_TASK||' + this.userId + '||c,' + this.perceivedControl + ',u,' + this.inputUncertainty + '\n');
      this.toggleModal();
      // Event listener to close the modal on Esc
      document.addEventListener("keydown", (e) => {
        if (this.showModal && e.keyCode == 27) {
          this.closeModal()
        }
      });

      //this.maxTopic = this.findMaxTopic()
    }

  }, //End methods
});
