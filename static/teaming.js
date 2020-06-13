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
        sliderValue: 4, // init the slider to the middle
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
        modelAccuracy: 0,
        logText: '',
        firstUpdate: true, // track whether its the first load of data (treated differently then later saves)
        startDate: null,
        timer: null,
        totalTime: 30 * 60 * 1000, // total time is 30 minutes
        taskTime: 0,
        //  time: 0, // initially, time is 0
        paused: false, // track when the user is on the instructions or alert page (at which time we pause the task)
        timeWarning: false, // track whether the user should see the time warning alert
        timeWarningFifteen: false,
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
            let seconds = parseInt(this.taskTime / 1000);
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
        /**
         * Method fires when the slider is changed. 
         * @param {*} ev 
         */
        sliderChange: function (ev) {
            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'adherence': this.sliderValue,
                'activity': 'adherenceChange'
            };
            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log('LOGGED:', JSON.stringify(log));
            for (i = 0; i < this.unlabeledDocs.length; i++) {
                let doc = this.unlabeledDocs[i];
                if (doc.hasOwnProperty('userLabel')) {
                    label = doc.userLabel;
                    this.computedProjectedClassification(doc, label, this.sliderValue);
                }
            }

        },
        startTask: function () {
            this.startDate = new Date();

            // get a new user
            //this.getNewUser();

            // INITIAL UPDATE
            // comment out the below to hide the tutorial
            this.sendUpdate();
            this.finished = false;
            this.paused = false;
            this.taskTime = this.totalTime;

            // two minutes remaining warning
            this.twoMinute = setTimeout(() => {
                log = {
                    'user': this.userId,
                    'currTime': this.getCurrTimeStamp(),
                    'activeTime': this.getActiveTime(),
                    'activity': 'time2'
                };
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));
                // show in modal and pause task time
                this.timeWarning = true;
                this.openModal();
            }, this.totalTime - (2 * 60 * 1000));

            // fifteen minutes remaining warning
            this.twoMinute = setTimeout(() => {
                log = {
                    'user': this.userId,
                    'currTime': this.getCurrTimeStamp(),
                    'activeTime': this.getActiveTime(),
                    'activity': 'time15'
                };
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));
                // show in modal and pause task time
                this.timeWarningFifteen = true;
                this.openModal();
            }, this.totalTime - (15 * 60 * 1000));

            // set a task timer for 30 minutes (this.taskTime)
            this.timer = setInterval(() => {
                if (this.taskTime > 0) {
                    // count down the timer 
                    this.taskTime -= 1000;
                } else {
                    clearInterval(this.timer);
                    log = {
                        'user': this.userId,
                        'currTime': this.getCurrTimeStamp(),
                        'activeTime': this.getActiveTime(),
                        'activity': 'time0'
                    };
                    this.logText += JSON.stringify(log);
                    this.logText += '\n';
                    console.log('LOGGED:', JSON.stringify(log));
                    // send final log
                    this.saveLog();
                    // set finished status to true
                    this.finished = true;
                    // open the modal
                    this.openModal();
                }
            }, 1000);

            this.showModal = false;
            this.started = true;

            log = {
                'user':this.userId,
                'currTime':this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'activity': 'startTask',
                'uncertainty': this.inputUncertainty,
                'adherence': this.sliderValue
            };

            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log('LOGGED:', JSON.stringify(log));

            // close the modal
            //this.toggleModal();

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
            if (id === '') {
                console.error('no user id provided');
                return;
            }
            axios.get('/api/checkuserid/' + id).then(response => {
                console.warn('checking user id', id);
                this.checked = response.data.hasId;
                if (response.data.hasId) {
                    this.userId = id;
                    this.sendUpdate();
                }
                else {
                    console.error('That user id was not found', id);
                }
            }).catch(error => {
                console.error('error in /api/checkuserid', error)
            });
        },
        /**
         * Method to register a new user for the task; returns the userId 
         */
        getNewUser: function () {
            axios.post('/api/adduser').then(response => {
                this.userId = response.data.userId;
                log = {
                    'user': this.userId,
                    'currTime': this.getCurrTimeStamp(),
                    'activity': 'loadTool',
                    'uncertainty': this.inputUncertainty
                };
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));
                // include the below to hide the tutorial
                // this.sendUpdate();
            }).catch(error => {
                console.error('error in /api/adduser', error);
            });
        },

        save: function () {
            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activity': 'clickedSave'
            };
            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log("LOGGED:", JSON.stringify(log));
            this.sendUpdate();
        },

        saveLog: function() {
            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'activity': 'endTask'
            };

            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log("LOGGED:", JSON.stringify(log));

            axios.post('/api/update', {
                labeled_docs: [],
                user_id: this.userId,

                // updates the log text on call to update
                log_text: this.logText,
                desired_adherence: this.sliderValue,
            })
        },

        sendUpdate: function () {
            if (this.finished) {
                return;
            }

            this.loading = true;
            var curLabeledDocs = this.unlabeledDocs.filter(
                doc => doc.hasOwnProperty('userLabel'))
                .map(doc => ({
                    doc_id: doc.docId,
                    user_label: doc.userLabel
                }));
            this.labeledCount += curLabeledDocs.length;

            if (curLabeledDocs.length > 0) {
                this.inputProvided = true;

            } else {
                this.inputProvided = false;

            }

            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'activity': 'saveLabels',
                'labels': [],
                'totalLabeled': this.labeledCount,
                'labeled': curLabeledDocs.length,
                'adherence': this.sliderValue,
                'condition': this.inputUncertainty
            };

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

                // doc id, true label, system label, system label confidence, user label, highlights
                log.labels.push({
                    'id':d.docId,
                    'label': (d.hasOwnProperty('userLabel') ? d.userLabel : 'Unlabeled')
                });
            }

            log['correctLabels'] = correctLabels;
            log['incorrectLabels'] = incorrectLabels;


            // if it's not the first update, write the log
            if(!this.firstUpdate) {
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));
            }
            
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

                // track updated model accuracy
                this.modelAccuracy = response.data.modelAccuracy;
                log = {
                    'user': this.userId,
                    'currTime': this.getCurrTimeStamp(),
                    'activeTime': this.getActiveTime(),
                    'activity': 'updatedModel',
                    'documents': [],
                    'modelAccuracy': response.data.modelAccuracy
                };

                // log info for each new item
                for (var i = 0; i < response.data.unlabeledDocs.length; i++) {
                    doc = response.data.unlabeledDocs[i];
                    d = {
                        'id': doc.docId,
                        'text': doc.text,
                        'highlights': doc.highlights,
                        'trueLabel': doc.trueLabel,
                        'confRep': Math.round(doc.prediction.confidence * 100),
                        'confDem': Math.round((1 - doc.prediction.confidence) * 100),
                        'predictedLabel': doc.prediction.label
                    };
                    log.documents.push(d);
                }
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));

                this.labels = response.data.labels;
                this.labeled_docs = [];
                this.loading = false;
                this.isMounted = true;

                // set the dem/rep colors
                if (!this.colorsChosen) {
                    this.chooseColors();
                    this.colorsChosen = true;
                }

                // process the formatted html for each doc
                this.unlabeledDocs.forEach(function (doc) {
                    this.setDocHtml(doc);
                }, this);


                for (var i = 0; i < this.unlabeledDocs.length; i++) {
                    Vue.set(this.unlabeledDocs[i], 'open', true);
                }

                // pop up the modal (but not on first update)
                if (!this.firstUpdate) {
                    this.refreshed = true;
                    this.openModal();
                } else {
                    this.firstUpdate = false;
                }
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
            // if the task has started we want to log that the instructions were closed
            if (this.started) {
                this.timeWarning = false;
                this.timeWarningFifteen = false;
                this.paused = false;
                this.showModal = false;
                this.refreshed = false;
                log = {
                    'user': this.userId,
                    'currTime': this.getCurrTimeStamp(),
                    'activeTime': this.getActiveTime(),
                    'activity': 'closeInstructions'
                };
                this.logText += JSON.stringify(log);
                this.logText += '\n';
                console.log('LOGGED:', JSON.stringify(log));
            }
        },
        openModal: function () {
            this.paused = true;
            this.showModal = true;
            this.modalState = 0;
            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'activity': 'openInstructions'
            };
            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log('LOGGED:', JSON.stringify(log));
        },
        toggleModal: function () {
            if (this.showModal) {
                this.closeModal()
            } else {
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
            
          //  anything = '[^a-zA-Z]+'
            anything = /^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]*$/
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
        setDocHtml: function (doc) {
            var html = doc.text;
            // remove the url token
          //  var html = doc.text.replace("&lt;URL_TOKEN&gt;", "");
          //  html = html.replace("<URL_TOKEN>", "");
            var prev = 0
            var loc;
            var label;
            var a;
            var b;
            var offsets = [];
            var start = 0;
            var end = 0;
            for (var i = 0; i < doc.highlights.length; i++) {
                var ngram = doc.highlights[i][0];
                var label = doc.highlights[i][1];

                // convert to regex so we're only highlighting words (not matching substrings of other words)
                var ngrams_regex = this.convertToRegex(ngram.split(' '));
              //  var re = new RegExp(ngrams_regex, 'g');
                //    console.log(html.substring(end));
             //   var re = ' ' + ngram + ' ';
             //   console.log(re);
                var start = html.substring(end).search(' ' + ngram + ' ');
                if (start == -1) {
                   start = html.substring(end).search(ngram + ' ');
                    if (start == -1) {
                        start = html.substring(end).search(' ' + ngram);
                        if (start == -1) {
                            start = html.substring(end).search(ngram);
                        } else {
                            start = start + 1;
                        }
                    }
                } else {
                    start = start + 1;
                }
                console.log('match index', start + end);
                    if (start !== -1) {
                        start = start + end;
                        var end = start + ngram.length;
                        //       console.log(ngram, start, end);
                        offsets.push([start, end, this.colors[label]]);
                    } else {
                        console.warn('unmatched highlighted token', ngram, "not found in", html);
                    }





                //  html = html.replace(re, '$1<span class="rounded" style="background-color: ' + doc_label + '">$2</span>$3');
                //  console.log(html);
            }

            //   console.log(offsets);
            // iterate over the offsets from end to start and add in the spans
            for (var i = offsets.length - 1; i >= 0; i--) {
                //     console.log(html);
                html = html.slice(0, offsets[i][1]) + "</span>" + html.slice(offsets[i][1]);
                html = html.slice(0, offsets[i][0]) + "<span class='rounded' style='background-color:" + offsets[i][2] + "'>" + html.slice(offsets[i][0]);
                //    console.log(html);
            }

            doc.formattedHtml = html;
        },
        deleteLabel: function (doc) {
            Vue.delete(doc, 'userLabel');
        },
        pad: function (num, size) {
            return ('000000' + num).substr(-size);
        },
        /**
         * Method to convert the backend label into human readable abbreviation
         * @param {} label 
         */
        expandLabel: function (label) {
            if (label === 'D') return 'Dem';
            if (label === 'R') return 'Rep';
        },
        /**
         * Method to distinguish possibly vs. probably dem and republican colors
         * @param {*} color 
         * @param {*} val 
         */
        lightenDarkenColor: function (color, val) {
            if (typeof (color) === 'string') {
                var rgb = hexToRgb(color);
                rgb = [rgb.r, rgb.g, rgb.b];
            }
            for (var i = 0; i < 3; i++) {
                rgb[i] -= val;
                rgb[i] = Math.min(255, Math.max(0, rgb[i]));
            }
            return '#' + (rgb[2] | rgb[1] << 8 | rgb[0] << 16).toString(16);
        },
        /**
         * computes the democrat bar width
         * @param {} doc 
         */
        updateWidth: function (doc) {
            // TODO: confirm assumption that confidence always refers to republican percentage
            w = Math.round((1 - doc.prediction.confidence) * 100);
            return "width:" + w + "%;"
        },
        labelDoc: function (doc, label) {
            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'doc': {
                    'id':doc.docId,
                    'text': doc.text,
                    'highlights': doc.highlights,
                    'trueLabel': doc.trueLabel,
                    'confRep': Math.round(doc.prediction.confidence * 100),
                    'confDem': Math.round((1 - doc.prediction.confidence) * 100),
                    'predictedLabel': doc.prediction.label
                }
            };

            if (doc.hasOwnProperty('userLabel')) {
                // unlabel
                if (doc.userLabel === label) {
                    this.deleteLabel(doc);
                    log['activity'] = "unlabelDoc";
                    this.logText += JSON.stringify(log);
                    this.logText += '\n';
                    console.log('LOGGED:', JSON.stringify(log));
                    doc.updated = false;
                    return;
                }
            }
            Vue.set(doc, 'userLabel', label);
            log['activity'] = 'labelDoc';
            log['label'] = label;
            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log('LOGGED:', JSON.stringify(log));

            // get the expected prediction
            //axio.get("get_expected_prediction(doc, desired_adherence, label, input_uncertainty):
            this.computedProjectedClassification(doc, label, this.sliderValue);
        },
        computedProjectedClassification: function (doc, label, adherence) {

            if (label === 'R2') {
                // possibly rep
                doc.projectedRep = Math.round(doc.expected_predictions.republican.possibly[adherence - 1] * 100);
                doc.projectedDem = 100 - doc.projectedRep;

            } else if (label === 'R1') {
                // probably rep
                doc.projectedRep = Math.round(doc.expected_predictions.republican.probably[adherence - 1] * 100);
                doc.projectedDem = 100 - doc.projectedRep;


            } else if (label === 'D2') {
                // possibly dem
                doc.projectedRep = Math.round(doc.expected_predictions.democrat.possibly[adherence - 1] * 100);
                doc.projectedDem = 100 - doc.projectedRep;

            } else if (label === 'D1') {
                // probably dem
                doc.projectedRep = Math.round(doc.expected_predictions.democrat.probably[adherence - 1] * 100);
                doc.projectedDem = 100 - doc.projectedRep;

            }

            log = {
                'user': this.userId,
                'currTime': this.getCurrTimeStamp(),
                'activeTime': this.getActiveTime(),
                'activity': 'updatedProjection',
                'doc': {
                    'id': doc.docId,
                    'confRep': Math.round(doc.prediction.confidence * 100)
                },
                'label': label,
                'adherence': adherence,
                'projectedConfRep': doc.projectedRep
            };

            this.logText += JSON.stringify(log);
            this.logText += '\n';
            console.log('LOGGED:', JSON.stringify(log));
            doc.updated = true;
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
        getExactTime: function () {
            return new Date() - this.startDate;
        },
        getActiveTime: function () {
            if (this.taskTime === 0) {
                return -1;
            } else {
                return (this.totalTime - this.taskTime) / 1000;
            }
        },
        getCurrTimeStamp: function () {
            return new Date();
        },

        getDemConfidence: function (doc) {
            // TODO: confirm that the doc.prediction.confidence is ALWAYS the % chance of being republican
            dcon = Math.round((1 - doc.prediction.confidence) * 100);
            return dcon;
        },

        getRepConfidence: function (doc) {
            rcon = Math.round(doc.prediction.confidence * 100);
            return rcon;
        }
    }, //End methods
});
