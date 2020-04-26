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
        vocab: [], // Corpus vocabulary 
        colors: {}, // TODO figure out what to do with this
        selectedDoc: {},
        loading: false,
        maxTopic: 1.000,
        isMounted: false,
        colorsChosen: false,
        showModal: true,
        autocompleteInput: '',
        autocompleteResults: [],
        userId: '',
        inputId: '',
        sliderValue: 0,
        sliderData: [1, 2, 3, 4, 5, 6, 7],
        showAnswers: false,
        showAnchorInfo: false,
        canEditAnchors: false,
        showTokens: false,
        displayInstructions: false,
        numCorrect: 0, // track the total number of correctly predicted documents the user was exposed to
        totalDocs: 0, // track the total number of predicted documents the user was exposed to
        // TODO: currently randomly choosing these conditions, but need to ensure that we get equal numbers in all conditions, so instead should use server to track how many participants of each condition
        // if input uncertainty is true, that means it's the four option condition
        inputUncertainty: Math.random() >= 0.5,
        labeledCount: 0,
        correctDocumentDelta: 0,
        logText: '',
        startDate: null,
        timer: null,
        totalTime: 20 * 60 * 1000, // total time is 20 minutes
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
    components: {
        VueSlider: window['vue-slider-component']
    },
    mounted: function () {
        console.log('app mounted');
        // get a new user
        this.getNewUser();
    }, //End mounted
    computed: {
        docsByLabel: function () {
            docsByLabelObj = {}
            for (var i = 0; i < this.labels.length; i++) {
                Vue.set(docsByLabelObj, this.labels[i], this.filterDocs(this.labels[i]));
            }

            return docsByLabelObj;
        },

        prettyTime: function () {
            let seconds = parseInt(this.time / 1000);
            let remaining = ('0' + (seconds % 60)).slice(-2);

            return parseInt(seconds / 60) + ':' + remaining;
        },

    }, //End computed
    watch: {}, //End watch
    methods: {

        // determine which tool screenshot to provide given the condition
        getScreenshotUrl: function () {
            if (this.inputUncertainty) {
                // assign and four options
                return '/static/images/assign-four-screenshot.png'
            } else {
                // assign and two options
                return '/static/images/assign-two-screenshot.png'
            }
        },
        sliderChange: function () {
            logString = (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||ADHERENCE_CHANGE||' + this.userId + '||' + this.sliderValue + '\n');
            this.logText += logString;
            console.log('LOGGED:', logString);

        },
        startTask: function () {
            console.log('starting the task!');
            this.startDate = new Date();

            // get a new user
            //this.getNewUser();

            // INITIAL UPDATE
            // comment out the below to hide the tutorial
            this.sendUpdate();
            this.finished = false;
            this.paused = false;
            this.time = this.totalTime;

            // two minutes remaining warning
            this.twoMinute = setTimeout(() => {
                this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||TIME_WARNING \n');
                // show in modal and pause task time
                this.timeWarning = true;
                this.openModal();
            }, this.totalTime - 2 * 60 * 1000);

            // task timer
            this.timer = setInterval(() => {
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
        },
        getVocab: function () {
            axios.get('/api/vocab').then(response => {
                this.vocab = response.data.vocab;
            }).catch(error => {
                console.error('error in /api/vocab', error)
            });
        },
        getIdData: function (id) {
            console.log('get id data', id);
            if (id === '') {
                alert('That user id was not found');
                return;
            }
            axios.get('/api/checkuserid/' + id).then(response => {
                logString = (response);

                this.logText += logString;
                console.log('LOGGED:', logString);
                this.checked = response.data.hasId;
                if (response.data.hasId) {
                    this.userId = id;
                    this.sendUpdate();
                }
                else {
                    this.logText += ("The user id was not found");
                    alert('That user id was not found');
                }
            }).catch(error => {
                console.error('error in /api/checkuserid', error)
            });
        },
        getNewUser: function () {
            console.log('get new user');
            axios.post('/api/adduser').then(response => {
                this.userId = response.data.userId;
                logString = (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||INITIAL_LOAD||' + this.userId + '||c,' + this.perceivedControl + ',u,' + this.inputUncertainty + '\n')
                this.logText += logString;
                console.log('LOGGED:', logString);
                // include the below to hide the tutorial
                this.sendUpdate();
            }).catch(error => {
                console.error('error in /api/adduser', error);
                // console.log(error);
            });
        },

        sendUpdate: function () {
            if (this.finished) {
                return;
            }
            logString = this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||SEND_UPDATE||labeled,'
            //  this.logText += ;
            // Data is expected to be sent to server in this form:
            // data = {anchor_tokens: [[token_str,..],...]
            //         labeled_docs: [{doc_id: number
            //                         user_label: label},...]
            //        }
            this.loading = true;
            var curLabeledDocs = this.unlabeledDocs.filter(
                doc => doc.hasOwnProperty('userLabel'))
                .map(doc => ({
                    doc_id: doc.docId,
                    user_label: doc.userLabel.slice(0, -1)
                }));
            this.labeledCount += curLabeledDocs.length;
            logString += curLabeledDocs.length + ',total,' + this.labeledCount + '||';

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
                logString += ('doc,' + d.docId + ',true,' + d.trueLabel + ',pred,' + d.prediction.label + ',conf,' + d.prediction.confidence + ',user,' + (d.hasOwnProperty('userLabel') ? d.userLabel : 'Unlabeled') + ',highlights,' + d.highlights.length + ',D,' + numD + ',R,' + numR + ';');
            }
            // number of correct labels, number of incorrect labels (for the user)
            logString += "||correct," + correctLabels + ',incorrect,' + incorrectLabels;
            logString += '\n';
            this.logText += logString;
            console.log('LOGGED:', logString);

            axios.post('/api/update', {
                labeled_docs: curLabeledDocs,
                user_id: this.userId,

                // updates the log text on call to update
                log_text: this.logText,
                desired_adherence: this.sliderValue,
            }).then(response => {
                this.updateData = response.data;
                this.anchors = response.data.anchors;

                // new set of unlabeled documents
                this.unlabeledDocs = response.data.unlabeledDocs;
                console.log('Slider Value is:' + this.sliderValue);

                // determine the classifier accuracy for the returned set of documents, and track classifier accuracy for all documents the user has been exposed to

                this.correctDocumentDelta = response.data.correctDocumentDelta

                logString = (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||NEW_DEBATES||' + this.userId + '||currentAccuracy,' + this.correctDocumentDelta + '\n');
                this.logText += logString;
                console.log('LOGGED:', logString);

                // AMR 5/24: shuffle the order randomly (needed for teaming study)
                this.unlabeledDocs = this.shuffle(this.unlabeledDocs);
                this.labels = response.data.labels;
                this.labeled_docs = [];
                this.loading = false;
                this.isMounted = true;

                if (!this.colorsChosen) {
                    this.chooseColors();
                    this.colorsChosen = true;
                }

                for (var i = 0; i < this.unlabeledDocs.length; i++) {
                    Vue.set(this.unlabeledDocs[i], 'open', true);
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

        getAccuracy: function () {
            this.loading = true;

            axios.post('/api/accuracy', {
                user_id: this.userId
            }).then(response => {
                this.logText += (response);
                if (response.data.accuracy) {
                    console.log('current accuracy', response.data.accuracy)
                    this.accuracy = response.data.accuracy;
                }

                this.loading = false;

            }).catch(error => {
                console.error('Error in /api/accuracy', error);
                this.loading = false;

            });
        },
        shuffle: function (array) {
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
        chooseColors: function () {

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
        closeModal: function () {
            if (this.started) {
                this.timeWarning = false;
                this.paused = false;
                this.showModal = false;
                this.refreshed = false;
                this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||CLOSE_INSTRUCTIONS \n');
            }
        },
        openModal: function () {
            this.paused = true;
            this.showModal = true;
            this.modalState = 0;
        },
        toggleModal: function () {
            if (this.showModal) {
                this.closeModal()
            } else {
                this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||OPEN_INSTRUCTIONS \n');
                this.openModal()
            }
        },
        filterDocs: function (label) {
            return this.docs.filter(function (doc) {
                return doc.label === label;
            });
        },
        labelAllCorrect: function () {
            for (var i = 0; i < this.unlabeledDocs.length; i++) {
                Vue.set(this.unlabeledDocs[i], 'userLabel', this.unlabeledDocs[i].trueLabel + '1');
            }
        },
        convertToRegex: function (ngrams) {
            anything = '[^a-zA-Z]+'
            //anything = '.*?'

            var regexBeginning = '(^|\\b)(';
            var fullRegex = regexBeginning;

            for (i = 0; i < ngrams.length; i++) {
                fullRegex += ngrams[i];
                if (i < (ngrams.length - 1)) {
                    fullRegex += `${anything}`;
                }
            }

            fullRegex += ')($|\\b)';

            return fullRegex;
        },
        getDocHtml: function (doc) {
            //    console.log('Getting HTML');
            var htmltext = doc.text;
            var html = htmltext.replace("&lt;URL_TOKEN&gt;", "");
            var prev = 0
            var loc;
            var label;
            var a;
            var b;
            for (var i = 0; i < doc.highlights.length; i++) {
                var ngram = doc.highlights[i][0];
                var label = doc.highlights[i][1];
                var ngrams_regex = this.convertToRegex(ngram.split(' '));
                var doc_label = this.colors[label];

                var re = new RegExp(ngrams_regex, 'g');
                html = html.replace(re, '$1<span class="rounded" style="background-color: ' + doc_label + '">$2</span>$3');

            }

            return html;
        },
        deleteLabel: function (doc) {
            Vue.delete(doc, 'userLabel');
        },
        labelDoc: function (doc, label) {
            if (doc.hasOwnProperty('userLabel')) {
                if (doc.userLabel === label) {
                    this.deleteLabel(doc);
                    this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||UNLABEL_DOC||' + doc.docId + '\n');
                    return;
                }
            }
            Vue.set(doc, 'userLabel', label);
            // timestamp, active time, label doc event, doc id, true label, system provided label, confidence, user provided label
            this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||LABEL_DOC||' + doc.docId + ',' + doc.trueLabel + ',' + doc.prediction.label + ',' + doc.prediction.confidence + ',' + label + '\n');
        },
        getConfidenceWord: function (doc) {
            // TODO: need a better way to set this threshold..
            return doc.prediction.confidence < .95 ? 'Possibly' : 'Probably';
        },
        getConfidenceColor: function (doc) {
            if (doc.prediction.confidence < .95) {
                return this.lightenDarkenColor(this.colors[doc.prediction.label], -40);
            } else {
                return this.colors[doc.prediction.label];
            }
        },
        toggleDocOpen: function (doc) {
            if (doc.open) {
                this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||CLOSE_DOC||' + doc.docId + '\n');
            } else {
                this.logText += (this.getCurrTimeStamp() + '||' + this.getActiveTime() + '||OPEN_DOC||' + doc.docId + '\n');
            }
            doc.open = !doc.open;
        },
        getExactTime: function () {
            return new Date() - this.startDate;
        },
        getActiveTime: function () {
            return this.totalTime - this.time;
        },
        getCurrTimeStamp: function () {
            return new Date();
        },

        getDemConfidence: function (doc) {
            dcon = Math.round(doc.prediction.label == "D" ? doc.prediction.confidence * 100 : (1 - doc.prediction.confidence) * 100);
            return dcon;
        },

        getRepConfidence: function (doc) {
            rcon = Math.round(doc.prediction.label == "R" ? doc.prediction.confidence * 100 : (1 - doc.prediction.confidence) * 100);
            return rcon;
        }
    }, //End methods
});
