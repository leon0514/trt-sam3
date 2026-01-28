document.addEventListener('DOMContentLoaded', () => {
    const state = {
        target: { img: new Image(), boxes: [], file: null },
        prompt: { img: new Image(), boxes: [], file: null },
        activeKey: null
    };

    const modeSelect = document.getElementById('mode-select');
    const sectionPromptImg = document.getElementById('section-prompt-img');
    const sectionTextPrompt = document.getElementById('section-text-prompt');
    const targetThumbBox = document.getElementById('target-thumb-box');

    // 1. 核心模式切换逻辑 (混合模式处理)
    modeSelect.onchange = () => {
        const mode = modeSelect.value;
        
        // 参考图区域：仅跨图参考模式显示
        sectionPromptImg.classList.toggle('hidden', mode !== 'from-image');
        
        // 文本提示区域：文本模式 和 混合模式 都显示
        sectionTextPrompt.classList.toggle('hidden', mode === 'from-image');

        // 目标图点击逻辑：
        // 文本模式 (multi-class) -> 仅上传
        // 标注模式 (box) / 混合模式 (mixed) -> 上传并进入标注
        targetThumbBox.style.opacity = "1"; 
    };

    const textContainer = document.getElementById('text-container');
    function addTextInput(val = '') {
        const div = document.createElement('div');
        div.className = 'text-item';
        div.innerHTML = `<input type="text" class="prompt-val" value="${val}"><button class="del-btn">&times;</button>`;
        div.querySelector('.del-btn').onclick = () => div.remove();
        textContainer.appendChild(div);
    }
    document.getElementById('add-text-btn').onclick = () => addTextInput();
    addTextInput('person');

    // 2. 图片上传与标注入口
    ['target', 'prompt'].forEach(key => {
        const box = document.getElementById(`${key}-thumb-box`);
        const input = document.getElementById(`${key}-file`);
        const reBtn = document.getElementById(`re-${key}`); // 获取重传按钮
        
        // 点击逻辑分流
        box.onclick = (e) => {
            // 如果点击的是“重新上传”按钮，触发文件选择
            if (e.target === reBtn) {
                input.value = ''; // 关键修复 1：清空 value 确保触发 change
                input.click();
                return;
            }

            const mode = modeSelect.value;
            // 如果已经有图片，且当前模式支持标注，则进入编辑器
            if (state[key].file && mode !== 'multi-class') {
                openEditor(key);
            } else {
                // 否则（没图或者是文本模式），触发文件选择
                input.value = ''; // 关键修复 1
                input.click();
            }
        };

        input.onchange = (e) => {
            const file = e.target.files[0]; 
            if(!file) return;

            state[key].file = file;
            // 关键修复 2：重新上传时清空旧的标注框，防止坐标错位
            state[key].boxes = []; 
            
            const reader = new FileReader();
            reader.onload = (ev) => {
                state[key].img.onload = () => {
                    drawThumb(key);
                    // 上传成功后，如果是重传，需要确保预览遮罩隐藏
                    document.getElementById(`${key}-mask`).classList.add('hidden');
                };
                state[key].img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
        };
    });

    // 3. 标注编辑器控制
    const eCanvas = document.getElementById('editor-canvas');
    const eCtx = eCanvas.getContext('2d');

    function openEditor(key) {
        state.activeKey = key;
        document.getElementById('editor-modal').classList.remove('hidden');
        const ratio = Math.min(window.innerWidth*0.85/state[key].img.naturalWidth, window.innerHeight*0.8/state[key].img.naturalHeight, 1);
        eCanvas.width = state[key].img.naturalWidth * ratio;
        eCanvas.height = state[key].img.naturalHeight * ratio;
        renderEditor();
    }

    const getPos = (e) => {
        const rect = eCanvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (state[state.activeKey].img.naturalWidth / eCanvas.width),
            y: (e.clientY - rect.top) * (state[state.activeKey].img.naturalHeight / eCanvas.height)
        };
    };

    let drawing = false, startPos = {};
    eCanvas.onmousedown = (e) => { drawing = true; startPos = getPos(e); };
    eCanvas.onmousemove = (e) => {
        if(!drawing) return;
        renderEditor();
        const curr = getPos(e);
        drawBox(eCtx, startPos, curr, document.querySelector('input[name="pt"]:checked').value, true);
    };
    window.onmouseup = (e) => {
        if(!drawing) return; drawing = false;
        const end = getPos(e);
        if(Math.abs(end.x - startPos.x) > 5) {
            state[state.activeKey].boxes.push({
                x1: Math.min(startPos.x, end.x), y1: Math.min(startPos.y, end.y),
                x2: Math.max(startPos.x, end.x), y2: Math.max(startPos.y, end.y),
                type: document.querySelector('input[name="pt"]:checked').value
            });
        }
        renderEditor();
    };

    function renderEditor() {
        eCtx.clearRect(0, 0, eCanvas.width, eCanvas.height);
        eCtx.drawImage(state[state.activeKey].img, 0, 0, eCanvas.width, eCanvas.height);
        const s = eCanvas.width / state[state.activeKey].img.naturalWidth;
        state[state.activeKey].boxes.forEach(b => {
            eCtx.strokeStyle = b.type === 'pos' ? '#10b981' : '#ef4444';
            eCtx.lineWidth = 2;
            eCtx.strokeRect(b.x1 * s, b.y1 * s, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s);
        });
    }

    function drawBox(ctx, p1, p2, type, isTemp) {
        ctx.strokeStyle = type === 'pos' ? '#10b981' : '#ef4444';
        ctx.setLineDash(isTemp ? [5, 5] : []);
        const s = eCanvas.width / state[state.activeKey].img.naturalWidth;
        ctx.strokeRect(p1.x * s, p1.y * s, (p2.x - p1.x) * s, (p2.y - p1.y) * s);
    }

    // 撤销与清空
    document.getElementById('undo-draw').onclick = () => { state[state.activeKey].boxes.pop(); renderEditor(); };
    document.getElementById('clear-draw').onclick = () => { if(confirm("清空吗？")){state[state.activeKey].boxes=[]; renderEditor();} };
    document.getElementById('close-editor').onclick = () => { drawThumb(state.activeKey); document.getElementById('editor-modal').classList.add('hidden'); };

    function drawThumb(key) {
        const c = document.getElementById(`${key}-thumb-canvas`);
        const ctx = c.getContext('2d');
        const img = state[key].img;
        c.width = 300; c.height = 168;
        const s = Math.min(c.width/img.naturalWidth, c.height/img.naturalHeight);
        const ox = (c.width-img.naturalWidth*s)/2, oy = (c.height-img.naturalHeight*s)/2;
        ctx.clearRect(0,0,c.width,c.height);
        ctx.drawImage(img, ox, oy, img.naturalWidth*s, img.naturalHeight*s);
        state[key].boxes.forEach(b => {
            ctx.strokeStyle = b.type==='pos' ? '#10b981' : '#ef4444';
            ctx.lineWidth = 3;
            ctx.strokeRect(ox+b.x1*s, oy+b.y1*s, (b.x2-b.x1)*s, (b.y2-b.y1)*s);
        });
    }

    const confRange = document.getElementById('conf-range');
    const confVal = document.getElementById('conf-val');
    confRange.oninput = () => {
        confVal.innerText = parseFloat(confRange.value).toFixed(2);
    };

    // 2. 更新 run-btn 的点击事件处理逻辑
    document.getElementById('run-btn').onclick = async () => {
        if(!state.target.file) return alert("请上传目标图像");
        
        const loader = document.getElementById('loader');
        const imgOut = document.getElementById('res-img');
        const dlLink = document.getElementById('dl-link');
        const placeholder = document.getElementById('placeholder');
        const maskEnable = document.getElementById('mask-enable').checked; // 获取 mask 开关状态

        loader.classList.remove('hidden'); 
        imgOut.classList.add('hidden');
        dlLink.classList.add('hidden');

        const fd = new FormData();
        const mode = modeSelect.value;
        let apiEndpoint = (mode === 'person-refine') ? '/process-person-refine' : '/process-image';
        
        // --- 注入通用参数 ---
        fd.append('mode', mode);
        fd.append('target_image', state.target.file);
        fd.append('confidence_threshold', confRange.value); // 注入置信度
        fd.append('return_mask', maskEnable);               // 注入是否开启 mask

        const texts = Array.from(document.querySelectorAll('.prompt-val'))
                        .map(i => i.value.trim())
                        .filter(v => v)
                        .join(',');

        if (mode === 'from-image') {
            if (!state.prompt.file) {
                loader.classList.add('hidden');
                return alert("跨图模式请上传参考图");
            }
            fd.append('prompt_image', state.prompt.file);
            fd.append('prompt_boxes', JSON.stringify(state.prompt.boxes));
        } else {
            if (texts) fd.append('text_prompts', texts);
            if (state.target.boxes.length > 0) {
                fd.append('target_boxes', JSON.stringify(state.target.boxes));
            }
        }

        try {
            const r = await fetch(apiEndpoint, { 
                method: 'POST', 
                body: fd 
            });

            if (!r.ok) {
                const errText = await r.text();
                throw new Error(errText || "服务器推理出错");
            }

            const blob = await r.blob();
            const url = URL.createObjectURL(blob);
            
            imgOut.src = url;
            imgOut.classList.remove('hidden');
            placeholder.classList.add('hidden');
            dlLink.classList.remove('hidden');
            
            dlLink.onclick = () => {
                const a = document.createElement('a');
                a.href = url;
                a.download = `trt_sam3_${mode}_${Date.now()}.jpg`;
                a.click();
            };
        } catch (e) {
            console.error(e);
            alert("推理失败: " + e.message);
        } finally {
            loader.classList.add('hidden');
        }
    };

    modeSelect.dispatchEvent(new Event('change'));
});