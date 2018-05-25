




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
    //docList: [{number: 1, values: [1,1,1,1]}, {number: 2, values: [2,2,2,2]}, {number: 3, values: [3,3,3,3]}, {number: 4, values: [4,4,4,4]},],
    //labels: [{name: "Label1", count:2},{name: "Label2", count:2},],
    docs: {},
    labels: [],
    topics: [],
    colors: {},
    drag: {},
    selectedDoc: {},
    selectedTopic: {},
    loading: false,
    maxTopic: 1.0,
    },
  mounted: function () {
    this.loading = true;
    var self = this;
    $.ajax({
     // url: '/dist',
      url: '/testDocs',
      method: 'GET',
      success: function (data){
        console.log(data);
        self.docs = data.docs;
        self.labels = data.labels;
        self.topics = data.topics;

        var colorsList = ['#0000FF', '#8B0000', '#228B22', '#4B0082', '#FFA500', '#008080', '#FF4500'];
        var lenColors = colorsList.length;
        for (var i=0; i<self.labels.length; i++){
          Vue.set(self.colors, self.labels[i], colorsList[i%lenColors]);
        }
        self.maxTopic = self.findMaxTopic()
        self.loading = false;
      },
      error: function(error){
        console.log(error);
        self.error = true;
        self.loading = false;
        alert('AJAX Error');
      }
    });
  },
  computed: {
    activeTodos: function(){
      return this.todos.filter(function(item){
        return !item.completed;
      });
    },
    docsByLabel: function(){
      docsByLabelObj = {}
      console.log(this.labels);
      for(var i=0; i<this.labels.length; i++){
        Vue.set(docsByLabelObj, this.labels[i], this.filterDocs(this.labels[i]));
      }
      return docsByLabelObj;
    },
  },
  methods: {
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
  },
});

//var app = new Vue({
//  el: '#',
//  data: {
//    todos: [],
//    message: '',
//    show: 'all',
//    drag: {},
//    },
//  computed: {
//    activeTodos: function(){
//      return this.todos.filter(function(item){
//        return !item.completed;
//      });
//    },
//    filteredTodos: function(){
//      if (this.show==='active')
//        return this.todos.filter(function(item){
//          return !item.completed;
//        });
//      if (this.show==='completed')
//        return this.todos.filter(function(item){
//          return item.completed;
//        });
//      return this.todos;
//    },
//  },
//  methods: {
//    addItem: function(){
//      this.todos.push({text: this.message, completed:false});
//      this.message = '';
//    },
//    completeItem: function(item){
//      item.completed = !item.completed;
//    },
//    deleteItem: function(item){
//      var index = this.todos.indexOf(item);
//      if (index>-1)
//        this.todos.splice(index,1);
//    },
//    showAll: function(){
//      this.show = 'all';
//    },
//    showActive: function(){
//      this.show = 'active';
//    },
//    showCompleted: function(){
//      this.show = 'completed';
//    },
//    deleteCompleted: function(){
//      this.todos = this.todos.filter(function(item){
//        return !item.completed;
//      });
//    },
//    dragItem: function(item){
//      this.drag = item;
//      //item
//    },
//    dropItem: function(item){
//      var indexItem = this.todos.indexOf(this.drag);
//      var indexTarget = this.todos.indexOf(item);
//      this.todos.splice(indexItem,1);
//      this.todos.splice(indexTarget,0,this.drag);
//    },
//  },
//});
