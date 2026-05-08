document.addEventListener('DOMContentLoaded', () => {
    // 状态管理
    const state = {
        target: { img: new Image(), boxes: [], file: null },
        prompt: { img: new Image(), boxes: [], file: null },
        activeKey: null
    };

    const modeSelect = document.getElementById('mode-select');
    const sectionPromptImg = document.getElementById('section-prompt-img');
    const sectionTextPrompt = document.getElementById('section-text-prompt');
    const sectionPredefined = document.getElementById('section-predefined-text');
    const sectionCropConfig = document.getElementById('section-crop-config');

    // 1. 模式切换逻辑
    modeSelect.onchange = () => {
        const mode = modeSelect.value;
        sectionPromptImg.classList.toggle('hidden', mode !== 'from-image');
        sectionTextPrompt.classList.toggle('hidden', mode === 'from-image');
        // 仅在 obj-refine 模式下显示预检测标签和 crop 配置区域
        sectionPredefined.classList.toggle('hidden', mode !== 'obj-refine');
        sectionCropConfig.classList.toggle('hidden', mode !== 'obj-refine');
    };

    // Crop 配置面板展开/折叠
    const cropConfigToggle = document.getElementById('crop-config-toggle');
    const cropConfigPanel = document.getElementById('crop-config-panel');
    cropConfigToggle.onclick = () => {
        const isHidden = cropConfigPanel.classList.contains('hidden');
        cropConfigPanel.classList.toggle('hidden', !isHidden);
        cropConfigToggle.innerText = isHidden ? '▲ 收起' : '▼ 展开';
    };

    // 2. 文本类别管理
    const textContainer = document.getElementById('text-container');
    function addTextInput(val = '') {
        const div = document.createElement('div');
        div.className = 'text-item';
        div.innerHTML = `
            <input type="text" class="prompt-val" value="${val}" placeholder="输入类别名称...">
            <button class="del-btn">&times;</button>
        `;
        div.querySelector('.del-btn').onclick = () => div.remove();
        textContainer.appendChild(div);
    }
    document.getElementById('add-text-btn').onclick = () => addTextInput();
    addTextInput('person');

    // 2.5 预检测标签管理
    const predefinedContainer = document.getElementById('predefined-container');
    function addPredefinedInput(val = '') {
        const div = document.createElement('div');
        div.className = 'text-item';
        div.innerHTML = `
            <input type="text" class="predefined-val" value="${val}" placeholder="输入预检测标签...">
            <button class="del-btn">&times;</button>
        `;
        div.querySelector('.del-btn').onclick = () => div.remove();
        predefinedContainer.appendChild(div);
    }
    document.getElementById('add-predefined-btn').onclick = () => addPredefinedInput();
    addPredefinedInput('person');

    // 3. 核心：图片上传与标注入口隔离修复
    ['target', 'prompt'].forEach(key => {
        const box = document.getElementById(`${key}-thumb-box`);
        const input = document.getElementById(`${key}-file`);
        const reBtn = document.getElementById(`re-${key}`);
        const mask = document.getElementById(`${key}-mask`);

        // A. 单独处理"重新上传"按钮 - 完全独立逻辑
        reBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            
            // 直接清空当前图片状态
            state[key].file = null;
            state[key].boxes = [];
            
            // 隐藏重新上传按钮
            reBtn.style.display = 'none';
            
            // 显示上传蒙版
            mask.classList.remove('hidden');
            
            // 立即触发文件选择
            input.value = '';
            input.click();
        });

        // B. 图片容器点击逻辑 - 只处理没有文件时的情况
        box.addEventListener('click', (e) => {
            // 如果点击的是"重新上传"按钮，已经处理过了，直接返回
            if (e.target.closest('.re-upload-btn')) {
                return;
            }
            
            // 如果有文件，打开编辑器
            if (state[key].file) {
                openEditor(key);
            } 
            // 如果没有文件，触发上传（但这里不应该发生，因为蒙版会覆盖整个区域）
            else {
                // 为了安全起见，也触发上传
                input.click();
            }
        });

        // C. 文件选择后的处理
        input.onchange = (e) => {
            const file = e.target.files[0]; 
            if (!file) return;

            // 更新状态
            state[key].file = file;
            state[key].boxes = [];
            
            // 创建图片对象
            const img = new Image();
            img.onload = () => {
                // 更新状态中的图片引用
                state[key].img = img;
                
                // 绘制缩略图
                drawThumb(key);
                
                // 隐藏上传蒙版
                mask.classList.add('hidden');
                
                // 显示重新上传按钮
                reBtn.style.display = 'block';
            };
            
            // 读取文件
            const reader = new FileReader();
            reader.onload = (ev) => {
                img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
        };
    });

    // 4. 标注编辑器逻辑
    const eCanvas = document.getElementById('editor-canvas');
    const eCtx = eCanvas.getContext('2d');

    function openEditor(key) {
        state.activeKey = key;
        const modal = document.getElementById('editor-modal');
        modal.classList.remove('hidden');
        
        const ratio = Math.min(
            (window.innerWidth * 0.85) / state[key].img.naturalWidth, 
            (window.innerHeight * 0.8) / state[key].img.naturalHeight, 
            1
        );
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
    eCanvas.onmousedown = (e) => { 
        drawing = true; 
        startPos = getPos(e); 
    };
    eCanvas.onmousemove = (e) => {
        if(!drawing) return;
        renderEditor();
        const curr = getPos(e);
        drawBox(eCtx, startPos, curr, document.querySelector('input[name="pt"]:checked').value, true);
    };
    window.onmouseup = (e) => {
        if(!drawing) return; 
        drawing = false;
        const end = getPos(e);
        if(Math.abs(end.x - startPos.x) > 5 || Math.abs(end.y - startPos.y) > 5) {
            state[state.activeKey].boxes.push({
                x1: Math.min(startPos.x, end.x), y1: Math.min(startPos.y, end.y),
                x2: Math.max(startPos.x, end.x), y2: Math.max(startPos.y, end.y),
                type: document.querySelector('input[name="pt"]:checked').value
            });
        }
        renderEditor();
    };

    function renderEditor() {
        if (!state.activeKey) return;
        eCtx.clearRect(0, 0, eCanvas.width, eCanvas.height);
        eCtx.drawImage(state[state.activeKey].img, 0, 0, eCanvas.width, eCanvas.height);
        const s = eCanvas.width / state[state.activeKey].img.naturalWidth;
        state[state.activeKey].boxes.forEach(b => {
            eCtx.strokeStyle = b.type === 'pos' ? '#10b981' : '#ef4444';
            eCtx.lineWidth = 2;
            eCtx.setLineDash([]);
            eCtx.strokeRect(b.x1 * s, b.y1 * s, (b.x2 - b.x1) * s, (b.y2 - b.y1) * s);
        });
    }

    function drawBox(ctx, p1, p2, type, isTemp) {
        ctx.strokeStyle = type === 'pos' ? '#10b981' : '#ef4444';
        ctx.setLineDash(isTemp ? [5, 5] : []);
        const s = eCanvas.width / state[state.activeKey].img.naturalWidth;
        ctx.strokeRect(p1.x * s, p1.y * s, (p2.x - p1.x) * s, (p2.y - p1.y) * s);
    }

    document.getElementById('undo-draw').onclick = () => { state[state.activeKey].boxes.pop(); renderEditor(); };
    document.getElementById('clear-draw').onclick = () => { if(confirm("清空吗？")){state[state.activeKey].boxes=[]; renderEditor();} };
    document.getElementById('close-editor').onclick = () => { 
        drawThumb(state.activeKey); 
        document.getElementById('editor-modal').classList.add('hidden'); 
    };

    // 5. 缩略图渲染
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

    // 6. UI 数值实时更新
    const confRange = document.getElementById('conf-range');
    const confVal = document.getElementById('conf-val');
    confRange.oninput = () => {
        confVal.innerText = parseFloat(confRange.value).toFixed(2);
    };

    // 7. 推理提交逻辑 (无改动)
    document.getElementById('run-btn').onclick = async () => {
        if(!state.target.file) return alert("请上传目标图像");
        
        const loader = document.getElementById('loader');
        const imgOut = document.getElementById('res-img');
        const dlLink = document.getElementById('dl-link');
        const placeholder = document.getElementById('placeholder');
        const maskEnable = document.getElementById('mask-enable').checked;
        const mergeEnable = document.getElementById('merge-results-enable').checked;

        loader.classList.remove('hidden'); 
        imgOut.classList.add('hidden');
        dlLink.classList.add('hidden');

        const fd = new FormData();
        const mode = modeSelect.value;
        let apiEndpoint = (mode === 'obj-refine') ? '/process-obj-refine' : '/process-image';
        
        fd.append('mode', mode);
        fd.append('target_image', state.target.file);
        fd.append('confidence_threshold', confRange.value); 
        fd.append('return_mask', maskEnable);               

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

        // obj-refine 模式需要额外传递预检测标签、合并开关和 crop 配置
        if (mode === 'obj-refine') {
            const preDefs = Array.from(document.querySelectorAll('.predefined-val'))
                              .map(i => i.value.trim())
                              .filter(v => v)
                              .join(',');
            if (preDefs) {
                fd.append('pre_defined_texts', preDefs);
            }
            fd.append('merge_results', mergeEnable);

            // 收集 crop 配置
            const cropCfg = {
                max_size: parseInt(document.getElementById('crop-max-size').value) || 640,
                padding: parseInt(document.getElementById('crop-padding').value) || 20,
                w_diou: parseFloat(document.getElementById('crop-w-diou').value) || 30.0,
                w_expansion: parseFloat(document.getElementById('crop-w-expansion').value) || 5.0,
                count_penalty: parseFloat(document.getElementById('crop-count-penalty').value) || 120.0,
                nms_threshold: parseFloat(document.getElementById('crop-nms-threshold').value) || 0.2,
                enable_ar_fix: document.getElementById('crop-enable-ar').checked,
                target_ar: parseFloat(document.getElementById('crop-target-ar').value) || 1.0
            };
            fd.append('crop_config_json', JSON.stringify(cropCfg));
        }

        try {
            const r = await fetch(apiEndpoint, { method: 'POST', body: fd });
            if (!r.ok) throw new Error(await r.text() || "服务器推理出错");

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

    // 初始化运行一次切换逻辑
    modeSelect.dispatchEvent(new Event('change'));
});