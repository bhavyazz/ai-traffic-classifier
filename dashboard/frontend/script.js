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
            const res = await fetch('http://127.0.0.1:8000/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Upload failed');
            
            renderResults(data, 'up');
            uploadResults.classList.remove('hidden');
        } catch (err) {
            alert(err.message);
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

    // Capture Logic
    const startCaptureBtn = document.getElementById('startCaptureBtn');
    const captureDuration = document.getElementById('captureDuration');
    const captureLoader = document.getElementById('captureLoader');
    const captureResults = document.getElementById('captureResults');
    const captureStatusText = document.getElementById('captureStatusText');
    const captureProgress = document.getElementById('captureProgress');

    // Create capture results structure if not exists
    if (captureResults.innerHTML.trim() === '') {
        captureResults.innerHTML = document.getElementById('uploadResults').innerHTML.replace(/up/g, 'cap');
    }

    startCaptureBtn.addEventListener('click', async () => {
        const duration = parseInt(captureDuration.value);
        
        startCaptureBtn.disabled = true;
        captureLoader.classList.remove('hidden');
        captureResults.classList.add('hidden');
        captureProgress.style.width = '0%';
        captureProgress.style.transition = `width ${duration}s linear`;
        
        // Force reflow
        void captureProgress.offsetWidth;
        captureProgress.style.width = '100%';
        
        let countdown = duration;
        captureStatusText.textContent = `Capturing network traffic... (${countdown}s)`;
        
        const timer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                captureStatusText.textContent = `Capturing network traffic... (${countdown}s)`;
            } else {
                captureStatusText.textContent = `Processing PCAP and classifying...`;
                clearInterval(timer);
            }
        }, 1000);

        try {
            const res = await fetch('http://127.0.0.1:8000/api/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ duration })
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            renderResults(data, 'cap');
            captureResults.classList.remove('hidden');
        } catch (err) {
            alert(err.message);
        } finally {
            startCaptureBtn.disabled = false;
            captureLoader.classList.add('hidden');
            captureProgress.style.transition = 'none';
            captureProgress.style.width = '0%';
            clearInterval(timer);
        }
    });

    // Analytics Logic
    async function loadAnalytics() {
        const grid = document.getElementById('analyticsGrid');
        try {
            const res = await fetch('http://127.0.0.1:8000/api/analytics');
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
