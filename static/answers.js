var app = new Vue({
  el: '#app',
  data: {
    userId: 'PMgSi',
    userDocs: [],
    isMounted: false,
  },
  methods: {
    getUserData: function(){
      if (this.userId === ''){
        alert('That user id was not found');
        return;
      }
      axios.get('/api/getuserdata/'+this.userId).then(response => {
        this.userDocs = response.data.documents;
        this.isMounted = true;
      }).catch(error => {
        console.log('error in /api/getuserdata');
        console.log(error);
      });
    },
    getCorrect: function(){
      var tot = 0;
      for (var i=0; i<this.userDocs.length; i++){
        if (this.userDocs[i].metadata.correct){
          tot++;
        }
      }
      return tot;
    },
  }, //End methods
});
