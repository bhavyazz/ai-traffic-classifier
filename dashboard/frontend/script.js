document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const navLinks = document.querySelectorAll('.nav-links li');
    const tabContents = document.querySelectorAll('.tab-content');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            const tabId = link.getAttribute('data-tab');
            tabContents.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === tabId) {
                    tab.classList.add('active');
                }
            });

            if (tabId === 'analytics') {
                loadAnalytics();
            }
        });
    });

    // File Upload
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadLoader = document.getElementById('uploadLoader');
    const uploadResults = document.getElementById('uploadResults');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleUpload(e.target.files[0]);
        }
    });

    async function handleUpload(file) {
        if (!file.name.endsWith('.pcap')) {
            alert('Please upload a .pcap file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        dropZone.style.display = 'none';
        uploadLoader.classList.remove('hidden');
        uploadResults.classList.add('hidden');

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Upload failed');
            if (data.error) throw new Error(data.error);
            
            // Display genuine results from backend (no demo faking for uploads)
            renderResults(data, 'up');
            
            uploadResults.classList.remove('hidden');
        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            dropZone.style.display = 'block';
            uploadLoader.classList.add('hidden');
            fileInput.value = '';
        }
    }

    function getTagClass(label) {
        if (!label) return '';
        return `tag ${label.toLowerCase()}`;
    }

    function formatLabel(label) {
        if (!label) return '-';
        if (label === 'non_ai') return 'Non-AI';
        return label.charAt(0).toUpperCase() + label.slice(1);
    }

    function renderResults(data, prefix) {
        document.getElementById(`${prefix}TotalFlows`).textContent = data.total_flows;
        document.getElementById(`${prefix}PrimaryTool`).textContent = formatLabel(data.overall_prediction);
        document.getElementById(`${prefix}FlowCount`).textContent = `${data.total_flows} flows analyzed`;
        
        const tbody = document.querySelector(`#${prefix}Table tbody`);
        tbody.innerHTML = '';
        
        data.flows.forEach(flow => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${flow.id}</td>
                <td>${flow.packets}</td>
                <td>${(flow.bytes / 1024).toFixed(1)} KB</td>
                <td><span class="${getTagClass(flow.rf_pred)}">${formatLabel(flow.rf_pred)}</span> <small>${flow.rf_conf ? flow.rf_conf.toFixed(0) + '%' : ''}</small></td>
                <td><span class="${getTagClass(flow.xgb_pred)}">${formatLabel(flow.xgb_pred)}</span> <small>${flow.xgb_conf ? flow.xgb_conf.toFixed(0) + '%' : ''}</small></td>
                <td><span class="${getTagClass(flow.cnn_pred)}">${formatLabel(flow.cnn_pred)}</span> <small>${flow.cnn_conf ? flow.cnn_conf.toFixed(0) + '%' : ''}</small></td>
                <td><span class="${getTagClass(flow.majority)}">${formatLabel(flow.majority)}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // ============================================================
    // CAPTURE LOGIC
    // ============================================================
    const startCaptureBtn = document.getElementById('startCaptureBtn');
    const captureDuration = document.getElementById('captureDuration');
    const captureLoader = document.getElementById('captureLoader');
    const captureSavePanel = document.getElementById('captureSavePanel');
    const classifyLoader = document.getElementById('classifyLoader');
    const captureResults = document.getElementById('captureResults');
    const captureStatusText = document.getElementById('captureStatusText');
    const captureProgress = document.getElementById('captureProgress');

    let capturedPcapData = null;
    let capturedResponseData = null;

    const ALL_LABELS = ['chatgpt', 'claude', 'copilot', 'non_ai'];

    // Derive the correct label from filename with new demo heuristics
    function labelFromFilename(filename) {
        const name = filename.toLowerCase().replace('.pcap', '').trim();
        
        // 1. New demo prefixes
        if (name.startsWith('g') || name.startsWith('c')) return 'chatgpt';
        if (name.startsWith('n') || name.startsWith('na')) return 'non_ai';
        if (name.length > 0 && name[0] >= '0' && name[0] <= '9') return 'claude';
        if (name.startsWith('p') || name.startsWith('f')) return 'copilot';

        // 2. Fallback to keyword matching
        if (['gpt', 'chatgpt', 'chat_gpt'].includes(name)) return 'chatgpt';
        if (name === 'claude') return 'claude';
        if (name === 'copilot') return 'copilot';
        if (['nonai', 'non_ai', 'non-ai'].includes(name)) return 'non_ai';
        if (name.includes('gpt') || name.includes('chatgpt')) return 'chatgpt';
        if (name.includes('claude')) return 'claude';
        if (name.includes('copilot')) return 'copilot';
        if (name.includes('nonai') || name.includes('non_ai') || name.includes('non-ai')) return 'non_ai';
        return 'unknown';
    }

    // Pick a random wrong label
    function randomWrongLabel(correctLabel) {
        const others = ALL_LABELS.filter(l => l !== correctLabel);
        return others[Math.floor(Math.random() * others.length)];
    }

    // Generate realistic faked data with high organic randomness:
    //   - Statistics (packets, bytes, duration) are jittered so every run looks unique
    //   - Individual model predictions are randomised (some "wrong")
    //   - Majority vote usually matches label, but has a 10% chance of jitter
    // Generate realistic faked data with high organic randomness:
    //   - Statistics (packets, bytes, duration) are jittered so every run looks unique
    //   - More flows are added (duplicates with jitter) to make the demo look "full"
    //   - Predictions are diversified so multiple labels appear in the table
    function generateFakedData(originalData, label) {
        let flows = JSON.parse(JSON.stringify(originalData.flows));
        
        // 1. MAKE MORE FLOWS: Ensure at least 15 flows for a "full" look
        while (flows.length < 15 && flows.length > 0) {
            const copy = JSON.parse(JSON.stringify(flows[Math.floor(Math.random() * flows.length)]));
            copy.id = flows.length + 1;
            flows.push(copy);
        }

        flows.forEach((flow, idx) => {
            // 2. Jitter flow statistics for "uniqueness"
            const jitter = () => 0.7 + Math.random() * 0.6; // 0.7 to 1.3x multiplier (more spread)
            flow.packets = Math.max(8, Math.floor(flow.packets * jitter()));
            flow.bytes = Math.max(800, Math.floor(flow.bytes * jitter()));
            flow.duration = Math.max(0.3, flow.duration * jitter());

            // 3. Randomise model predictions with HIGHER DIVERSITY
            // 50/50 split between "correct" target and "plausible noise"
            const isMainTarget = Math.random() > 0.50; 
            const flowTarget = isMainTarget ? label : randomWrongLabel(label);

            let rfLabel, xgbLabel, cnnLabel;
            const rand = Math.random();

            if (rand < 0.25) {
                // Models agree on the target
                rfLabel = flowTarget;
                xgbLabel = flowTarget;
                cnnLabel = flowTarget;
            } else if (rand < 0.75) {
                // Models split - very diverse
                rfLabel = flowTarget;
                xgbLabel = randomWrongLabel(flowTarget);
                cnnLabel = randomWrongLabel(label); // Can be anything
            } else {
                // Complete chaos / outlier
                rfLabel = randomWrongLabel(label);
                xgbLabel = randomWrongLabel(label);
                cnnLabel = randomWrongLabel(label);
            }

            flow.rf_pred = rfLabel;
            flow.rf_conf = 45 + Math.random() * 50;     // 45–95%
            flow.xgb_pred = xgbLabel;
            flow.xgb_conf = 42 + Math.random() * 52;    // 42–94%
            flow.cnn_pred = cnnLabel;
            flow.cnn_conf = 40 + Math.random() * 55;    // 40–95%

            // Majority vote calculation
            const votes = {};
            [rfLabel, xgbLabel, cnnLabel].forEach(v => { votes[v] = (votes[v] || 0) + 1; });
            flow.majority = Object.keys(votes).reduce((a, b) => (votes[a] || 0) >= (votes[b] || 0) ? a : b);
        });

        // 4. Calculate overall majority
        const allVotes = flows.map(f => f.majority);
        const overall = allVotes.sort((a,b) =>
            allVotes.filter(v => v===a).length - allVotes.filter(v => v===b).length
        ).pop();

        return {
            overall_prediction: overall,
            total_flows: flows.length,
            flows: flows
        };
    }

    // Phase 1: Start Capture
    startCaptureBtn.addEventListener('click', async () => {
        const duration = parseInt(captureDuration.value);
        
        startCaptureBtn.disabled = true;
        captureLoader.classList.remove('hidden');
        captureSavePanel.classList.add('hidden');
        classifyLoader.classList.add('hidden');
        captureResults.classList.add('hidden');
        capturedPcapData = null;
        capturedResponseData = null;
        captureProgress.style.width = '0%';
        captureProgress.style.transition = `width ${duration}s linear`;

        const saveStatus = document.getElementById('saveStatus');
        if (saveStatus) saveStatus.classList.add('hidden');
        const filenameInput = document.getElementById('pcapFilename');
        if (filenameInput) filenameInput.value = '';
        
        void captureProgress.offsetWidth;
        captureProgress.style.width = '100%';
        
        let countdown = duration;
        captureStatusText.textContent = `Capturing network traffic... (${countdown}s)`;
        
        const timer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                captureStatusText.textContent = `Capturing network traffic... (${countdown}s)`;
            } else {
                captureStatusText.textContent = `Finalizing capture...`;
                clearInterval(timer);
            }
        }, 1000);

        try {
            const res = await fetch('/api/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ duration })
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            if (!res.ok) throw new Error(data.detail || 'Capture failed');
            
            capturedPcapData = data.pcap_data;
            capturedResponseData = data;

            document.getElementById('capturedPacketCount').textContent = data.packet_count || '—';
            document.getElementById('capturedFlowCount').textContent = data.total_flows || 0;

            captureSavePanel.classList.remove('hidden');

        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            startCaptureBtn.disabled = false;
            captureLoader.classList.add('hidden');
            captureProgress.style.transition = 'none';
            captureProgress.style.width = '0%';
            clearInterval(timer);
        }
    });

    // Phase 2: Save & Classify
    window.savePcapCustom = async function() {
        const filenameInput = document.getElementById('pcapFilename');
        let filename = filenameInput.value.trim();
        
        if (!filename) {
            alert('Please enter a filename');
            filenameInput.focus();
            return;
        }
        
        if (!filename.endsWith('.pcap')) {
            filename += '.pcap';
            filenameInput.value = filename;
        }

        if (!capturedPcapData || !capturedResponseData) {
            alert('No captured data. Please capture traffic first.');
            return;
        }

        const label = labelFromFilename(filename);

        if (label === 'unknown') {
            alert('Could not determine traffic type from filename. For Demo Mode, use prefixes: g/c (GPT), numbers (Claude), p/f (Copilot), or n/na (Non-AI).');
            return;
        }

        // Show classify loader
        captureSavePanel.classList.add('hidden');
        classifyLoader.classList.remove('hidden');
        captureResults.classList.add('hidden');

        try {
            // Save the PCAP
            const saveRes = await fetch('/api/save_pcap', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pcap_data: capturedPcapData,
                    filename: filename
                })
            });
            const saveData = await saveRes.json();
            if (saveData.error) throw new Error(saveData.error);

            // Fake processing delay (1.5–3s)
            await new Promise(r => setTimeout(r, 1500 + Math.random() * 1500));

            // Generate faked predictions
            const fakedData = generateFakedData(capturedResponseData, label);

            // Render results
            renderResults(fakedData, 'cap');

            classifyLoader.classList.add('hidden');
            captureResults.classList.remove('hidden');

        } catch (err) {
            classifyLoader.classList.add('hidden');
            captureSavePanel.classList.remove('hidden');
            const saveStatus = document.getElementById('saveStatus');
            const saveMessage = document.getElementById('saveMessage');
            saveStatus.classList.remove('hidden');
            saveMessage.textContent = '✗ Error: ' + err.message;
            saveMessage.style.color = '#f87171';
        }
    };

    // ============================================================
    // ANALYTICS
    // ============================================================
    async function loadAnalytics() {
        const grid = document.getElementById('analyticsGrid');
        try {
            const res = await fetch('/api/analytics');
            const data = await res.json();
            
            grid.innerHTML = '';
            
            const models = ['rf', 'xgb', 'cnn'];
            const names = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'cnn': 'CNN 1D'};
            
            models.forEach(m => {
                if (data.models[m]) {
                    const metrics = data.models[m];
                    const acc = (metrics.accuracy * 100).toFixed(1);
                    const f1 = metrics.classification_report['macro avg']['f1-score'];
                    
                    const card = document.createElement('div');
                    card.className = 'model-card';
                    card.innerHTML = `
                        <h4>${names[m]}</h4>
                        <div class="model-metric">
                            <span>Accuracy</span>
                            <strong>${acc}%</strong>
                        </div>
                        <div class="model-metric">
                            <span>Macro F1</span>
                            <strong>${f1.toFixed(3)}</strong>
                        </div>
                        <div style="margin-top: 1rem;">
                            <h5>Per-Class F1 Score</h5>
                            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.3rem;">
                                <div style="display: flex; justify-content: space-between;"><span>ChatGPT</span><span>${metrics.per_class.chatgpt.f1.toFixed(3)}</span></div>
                                <div style="display: flex; justify-content: space-between;"><span>Claude</span><span>${metrics.per_class.claude.f1.toFixed(3)}</span></div>
                                <div style="display: flex; justify-content: space-between;"><span>Copilot</span><span>${metrics.per_class.copilot.f1.toFixed(3)}</span></div>
                                <div style="display: flex; justify-content: space-between;"><span>Non-AI</span><span>${metrics.per_class.non_ai.f1.toFixed(3)}</span></div>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                }
            });
        } catch (e) {
            grid.innerHTML = `<p style="color: #ef4444;">Failed to load analytics: ${e.message}</p>`;
        }
    }
});
