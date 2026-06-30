/* TRT SAM3 Web UI — 支持 multi-class / box / mixed / from-image / obj-refine */

// ===================== 状态 =====================
const state = {
    mode: 'multi-class',
    target: { file: null, image: null, boxes: [] },
    prompt: { file: null, image: null, boxes: [] },
    editorTarget: null, // 'target' | 'prompt'
};

let currentFile = null;
let lastResults = [];
let lastResultImage = null;
let activeLabels = new Set();
let displayConfidenceThreshold = 0.0;
let minAreaThreshold = 0;
let selectedResultIndex = null;
let checkedResultIndices = new Set();
let showOnlyChecked = false;
let lastVisibleItems = [];
let sortMode = 'default';

let tags = ['person'];
let predefinedTags = ['person'];

const colors = [
    '#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#a855f7',
    '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#6366f1'
];

// ===================== DOM 引用 =====================
const els = {
    modeSelect: document.getElementById('modeSelect'),

    dropZone: document.getElementById('dropZone'),
    imageInput: document.getElementById('imageInput'),
    targetThumbBox: document.getElementById('targetThumbBox'),
    previewImage: document.getElementById('previewImage'),
    editTargetBoxesBtn: document.getElementById('editTargetBoxesBtn'),
    reuploadTargetBtn: document.getElementById('reuploadTargetBtn'),
    targetBoxList: document.getElementById('targetBoxList'),

    promptImageGroup: document.getElementById('promptImageGroup'),
    promptDropZone: document.getElementById('promptDropZone'),
    promptImageInput: document.getElementById('promptImageInput'),
    promptThumbBox: document.getElementById('promptThumbBox'),
    promptPreviewImage: document.getElementById('promptPreviewImage'),
    editPromptBoxesBtn: document.getElementById('editPromptBoxesBtn'),
    reuploadPromptBtn: document.getElementById('reuploadPromptBtn'),
    promptBoxList: document.getElementById('promptBoxList'),

    tagList: document.getElementById('tagList'),
    tagAdd: document.getElementById('tagAdd'),
    tagInput: document.getElementById('tagInput'),
    tagInputLabel: document.getElementById('tagInputLabel'),
    modeHint: document.getElementById('modeHint'),

    predefinedLabelsGroup: document.getElementById('predefinedLabelsGroup'),
    predefinedTagList: document.getElementById('predefinedTagList'),
    predefinedTagAdd: document.getElementById('predefinedTagAdd'),

    mergeResultsGroup: document.getElementById('mergeResultsGroup'),
    mergeResults: document.getElementById('mergeResults'),

    cropConfigGroup: document.getElementById('cropConfigGroup'),
    cropConfigToggle: document.getElementById('cropConfigToggle'),
    cropConfigPanel: document.getElementById('cropConfigPanel'),
    cropMaxSize: document.getElementById('cropMaxSize'),
    cropPadding: document.getElementById('cropPadding'),
    cropWDiou: document.getElementById('cropWDiou'),
    cropWExpansion: document.getElementById('cropWExpansion'),
    cropCountPenalty: document.getElementById('cropCountPenalty'),
    cropNmsThreshold: document.getElementById('cropNmsThreshold'),
    cropTargetAr: document.getElementById('cropTargetAr'),
    cropEnableAr: document.getElementById('cropEnableAr'),

    confidence: document.getElementById('confidence'),
    confidenceValue: document.getElementById('confidenceValue'),
    showBoxes: document.getElementById('showBoxes'),
    showMasks: document.getElementById('showMasks'),
    showLabels: document.getElementById('showLabels'),
    returnMask: document.getElementById('returnMask'),
    submitBtn: document.getElementById('submitBtn'),
    stats: document.getElementById('stats'),

    sourceImage: document.getElementById('sourceImage'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    imageWrapper: document.getElementById('imageWrapper'),
    resultContainer: document.getElementById('resultContainer'),
    placeholder: document.getElementById('placeholder'),
    legend: document.getElementById('legend'),
    downloadBtn: document.getElementById('downloadBtn'),

    filterPanel: document.getElementById('filterPanel'),
    filterList: document.getElementById('filterList'),
    selectAllBtn: document.getElementById('selectAllBtn'),
    selectNoneBtn: document.getElementById('selectNoneBtn'),

    scoreFilterPanel: document.getElementById('scoreFilterPanel'),
    displayConfidence: document.getElementById('displayConfidence'),
    displayConfidenceValue: document.getElementById('displayConfidenceValue'),

    areaFilterPanel: document.getElementById('areaFilterPanel'),
    minAreaInput: document.getElementById('minArea'),

    resultListPanel: document.getElementById('resultListPanel'),
    sortBy: document.getElementById('sortBy'),
    detectionList: document.getElementById('detectionList'),
    toggleShowChecked: document.getElementById('toggleShowChecked'),
    clearCheckedBtn: document.getElementById('clearCheckedBtn'),

    jsonOutput: document.getElementById('jsonOutput'),
    toast: document.getElementById('toast'),

    editorModal: document.getElementById('editorModal'),
    editorTitle: document.getElementById('editorTitle'),
    editorCanvas: document.getElementById('editorCanvas'),
    closeEditor: document.getElementById('closeEditor'),
    undoBoxBtn: document.getElementById('undoBoxBtn'),
    clearBoxesBtn: document.getElementById('clearBoxesBtn'),
};

// ===================== 工具函数 =====================
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    return Math.abs(hash);
}

function getColor(label) {
    return colors[hashString(label) % colors.length];
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 255, g: 255, b: 255 };
}

function showToast(message, type = 'success') {
    els.toast.textContent = message;
    els.toast.className = 'toast ' + type;
    els.toast.style.display = 'block';
    setTimeout(() => { els.toast.style.display = 'none'; }, 3000);
}

function getBoxArea(item) {
    if (!item.box || item.box.length < 4) return 0;
    const [x1, y1, x2, y2] = item.box;
    return Math.max(0, (x2 - x1) * (y2 - y1));
}

function isResultVisible(item) {
    return activeLabels.has(item.label) &&
           item.score >= displayConfidenceThreshold &&
           getBoxArea(item) >= minAreaThreshold;
}

function updateStats() {
    const baseVisible = lastResults.filter(isResultVisible).length;
    const checkedVisible = lastResults.filter((item, idx) =>
        isResultVisible(item) && checkedResultIndices.has(idx)
    ).length;
    if (showOnlyChecked) {
        els.stats.textContent = `检测到 ${lastResults.length} 个目标，已勾选 ${checkedResultIndices.size} 个，当前展示 ${checkedVisible} 个`;
    } else {
        els.stats.textContent = `检测到 ${lastResults.length} 个目标，当前展示 ${baseVisible} 个，已勾选 ${checkedResultIndices.size} 个`;
    }
}

// ===================== 标签管理 =====================
function renderTags() {
    els.tagList.innerHTML = '';
    tags.forEach((tag, idx) => {
        const div = document.createElement('div');
        div.className = 'tag';
        div.innerHTML = `<span>${tag}</span><button type="button" data-idx="${idx}">&times;</button>`;
        els.tagList.appendChild(div);
    });
    els.tagList.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            tags.splice(parseInt(btn.dataset.idx), 1);
            renderTags();
        });
    });
}

function renderPredefinedTags() {
    els.predefinedTagList.innerHTML = '';
    predefinedTags.forEach((tag, idx) => {
        const div = document.createElement('div');
        div.className = 'tag';
        div.innerHTML = `<span>${tag}</span><button type="button" data-idx="${idx}">&times;</button>`;
        els.predefinedTagList.appendChild(div);
    });
    els.predefinedTagList.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            predefinedTags.splice(parseInt(btn.dataset.idx), 1);
            renderPredefinedTags();
        });
    });
}

els.tagAdd.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const val = els.tagAdd.value.trim();
        if (val && !tags.includes(val)) {
            tags.push(val);
            renderTags();
        }
        els.tagAdd.value = '';
    }
});

els.predefinedTagAdd.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const val = els.predefinedTagAdd.value.trim();
        if (val && !predefinedTags.includes(val)) {
            predefinedTags.push(val);
            renderPredefinedTags();
        }
        els.predefinedTagAdd.value = '';
    }
});

// ===================== 文件处理 =====================
function handleFile(file, targetKey) {
    if (!file) return;
    const obj = state[targetKey];
    obj.file = file;
    obj.boxes = [];
    obj.image = new Image();
    obj.image.onload = () => {
        renderThumb(targetKey);
        if (targetKey === 'target') currentFile = file;
    };
    const reader = new FileReader();
    reader.onload = (e) => { obj.image.src = e.target.result; };
    reader.readAsDataURL(file);
}

function renderThumb(targetKey) {
    const obj = state[targetKey];
    const imgEl = targetKey === 'target' ? els.previewImage : els.promptPreviewImage;
    const thumbBox = targetKey === 'target' ? els.targetThumbBox : els.promptThumbBox;
    const dropZone = targetKey === 'target' ? els.dropZone : els.promptDropZone;

    if (!obj.file || !obj.image) return;

    imgEl.src = obj.image.src;
    thumbBox.style.display = 'block';
    dropZone.style.display = 'none';
    renderBoxList(targetKey);
}

function resetUpload(targetKey) {
    const obj = state[targetKey];
    obj.file = null;
    obj.image = null;
    obj.boxes = [];
    const imgEl = targetKey === 'target' ? els.previewImage : els.promptPreviewImage;
    const thumbBox = targetKey === 'target' ? els.targetThumbBox : els.promptThumbBox;
    const dropZone = targetKey === 'target' ? els.dropZone : els.promptDropZone;
    const input = targetKey === 'target' ? els.imageInput : els.promptImageInput;

    imgEl.src = '';
    thumbBox.style.display = 'none';
    dropZone.style.display = 'block';
    input.value = '';
    if (targetKey === 'target') currentFile = null;
}

function setupDropZone(dropZone, input) {
    dropZone.addEventListener('click', () => input.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    });
}

setupDropZone(els.dropZone, els.imageInput);
els.imageInput.addEventListener('change', (e) => handleFile(e.target.files[0], 'target'));
els.reuploadTargetBtn.addEventListener('click', () => resetUpload('target'));
els.editTargetBoxesBtn.addEventListener('click', () => openEditor('target'));
els.previewImage.addEventListener('click', () => openEditor('target'));

setupDropZone(els.promptDropZone, els.promptImageInput);
els.promptImageInput.addEventListener('change', (e) => handleFile(e.target.files[0], 'prompt'));
els.reuploadPromptBtn.addEventListener('click', () => resetUpload('prompt'));
els.editPromptBoxesBtn.addEventListener('click', () => openEditor('prompt'));
els.promptPreviewImage.addEventListener('click', () => openEditor('prompt'));

// ===================== Box 列表渲染 =====================
function renderBoxList(targetKey) {
    const obj = state[targetKey];
    const container = targetKey === 'target' ? els.targetBoxList : els.promptBoxList;
    container.innerHTML = '';
    if (!obj.boxes.length) {
        container.textContent = '尚未标注 Box';
        return;
    }
    obj.boxes.forEach((box, idx) => {
        const div = document.createElement('div');
        div.className = 'box-list-item';
        const color = box.type === 'pos' ? '#22c55e' : '#ef4444';
        const coords = `[${Math.round(box.x1)}, ${Math.round(box.y1)}, ${Math.round(box.x2)}, ${Math.round(box.y2)}]`;
        div.innerHTML = `
            <span><span style="color:${color}">●</span> ${box.type} ${coords}</span>
            <button type="button" data-idx="${idx}">&times;</button>
        `;
        div.querySelector('button').addEventListener('click', () => {
            obj.boxes.splice(idx, 1);
            renderBoxList(targetKey);
        });
        container.appendChild(div);
    });
}

// ===================== 模式切换 UI =====================
const MODE_HINTS = {
    'multi-class': '输入一个或多个类别名称，后端会对每个类别单独推理。',
    'box': '在目标图上拖拽标注 positive / negative Box，模型根据几何框进行分割。需要后端配置 geometry-encoder。',
    'mixed': '同时输入文本标签并在目标图上标注 Box，结合两种提示进行推理。需要后端配置 geometry-encoder。',
    'from-image': '上传参考图并在参考图上标注 Box，模型把参考图的几何特征迁移到目标图。需要后端配置 geometry-encoder。',
    'obj-refine': '先通过"预检测标签"定位大致区域并裁剪，再用"精细检测标签"在裁剪区域做精细识别。'
};

function updateModeUI() {
    state.mode = els.modeSelect.value;

    // 参考图只在 from-image 显示
    els.promptImageGroup.style.display = state.mode === 'from-image' ? 'block' : 'none';

    // 目标图 Box 标注只在 box / mixed 使用；from-image 在参考图上标注
    const needTargetBoxes = ['box', 'mixed'].includes(state.mode);
    els.editTargetBoxesBtn.style.display = needTargetBoxes ? 'block' : 'none';

    // 文本标签在 multi-class / mixed / obj-refine 使用
    const needTextLabels = ['multi-class', 'mixed', 'obj-refine'].includes(state.mode);
    els.tagInput.parentElement.style.display = needTextLabels ? 'block' : 'none';

    // obj-refine 配置
    const isObjRefine = state.mode === 'obj-refine';
    els.predefinedLabelsGroup.style.display = isObjRefine ? 'block' : 'none';
    els.mergeResultsGroup.style.display = isObjRefine ? 'block' : 'none';
    els.cropConfigGroup.style.display = isObjRefine ? 'block' : 'none';

    // 标签输入框的文案
    if (isObjRefine) {
        els.tagInputLabel.textContent = '精细检测标签（按 Enter 添加）';
        els.tagAdd.placeholder = '输入精细检测标签后回车';
    } else if (state.mode === 'box') {
        els.tagInputLabel.textContent = '识别标签（可选，当前模式不使用）';
        els.tagAdd.placeholder = 'box 模式不需要文本标签';
    } else {
        els.tagInputLabel.textContent = '识别标签（按 Enter 添加）';
        els.tagAdd.placeholder = '输入标签后回车';
    }

    // 模式提示
    els.modeHint.textContent = MODE_HINTS[state.mode] || '';
    els.modeHint.className = ['box', 'mixed', 'from-image'].includes(state.mode)
        ? 'mode-hint warning'
        : 'mode-hint';
}

els.modeSelect.addEventListener('change', updateModeUI);

// ===================== Box 标注编辑器 (Modal) =====================
let editorDrawing = false;
let editorStartPos = null;

function openEditor(targetKey) {
    const obj = state[targetKey];
    if (!obj.image) {
        showToast('请先上传图片', 'error');
        return;
    }
    state.editorTarget = targetKey;
    els.editorTitle.textContent = targetKey === 'target' ? '标注目标图 Box' : '标注参考图 Box';
    els.editorModal.style.display = 'flex';

    const maxW = window.innerWidth * 0.85;
    const maxH = window.innerHeight * 0.65;
    const ratio = Math.min(maxW / obj.image.naturalWidth, maxH / obj.image.naturalHeight, 1);
    els.editorCanvas.width = obj.image.naturalWidth * ratio;
    els.editorCanvas.height = obj.image.naturalHeight * ratio;

    renderEditor();
}

function closeEditor() {
    els.editorModal.style.display = 'none';
    state.editorTarget = null;
    editorDrawing = false;
    editorStartPos = null;
}

function getEditorPos(e) {
    const rect = els.editorCanvas.getBoundingClientRect();
    const obj = state[state.editorTarget];
    const scaleX = obj.image.naturalWidth / els.editorCanvas.width;
    const scaleY = obj.image.naturalHeight / els.editorCanvas.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function renderEditor() {
    const targetKey = state.editorTarget;
    if (!targetKey) return;
    const obj = state[targetKey];
    const ctx = els.editorCanvas.getContext('2d');
    ctx.clearRect(0, 0, els.editorCanvas.width, els.editorCanvas.height);
    ctx.drawImage(obj.image, 0, 0, els.editorCanvas.width, els.editorCanvas.height);

    const sx = els.editorCanvas.width / obj.image.naturalWidth;
    const sy = els.editorCanvas.height / obj.image.naturalHeight;

    obj.boxes.forEach(box => {
        ctx.strokeStyle = box.type === 'pos' ? '#22c55e' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(box.x1 * sx, box.y1 * sy, (box.x2 - box.x1) * sx, (box.y2 - box.y1) * sy);
    });
}

els.editorCanvas.addEventListener('mousedown', (e) => {
    editorDrawing = true;
    editorStartPos = getEditorPos(e);
});

els.editorCanvas.addEventListener('mousemove', (e) => {
    if (!editorDrawing || !editorStartPos) return;
    renderEditor();
    const curr = getEditorPos(e);
    const ctx = els.editorCanvas.getContext('2d');
    const obj = state[state.editorTarget];
    const sx = els.editorCanvas.width / obj.image.naturalWidth;
    const sy = els.editorCanvas.height / obj.image.naturalHeight;

    ctx.strokeStyle = document.querySelector('input[name="boxType"]:checked').value === 'pos' ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    const x = Math.min(editorStartPos.x, curr.x) * sx;
    const y = Math.min(editorStartPos.y, curr.y) * sy;
    const w = Math.abs(curr.x - editorStartPos.x) * sx;
    const h = Math.abs(curr.y - editorStartPos.y) * sy;
    ctx.strokeRect(x, y, w, h);
});

window.addEventListener('mouseup', (e) => {
    if (!editorDrawing || !editorStartPos) return;
    editorDrawing = false;
    const end = getEditorPos(e);
    if (Math.abs(end.x - editorStartPos.x) > 5 && Math.abs(end.y - editorStartPos.y) > 5) {
        const type = document.querySelector('input[name="boxType"]:checked').value;
        const box = {
            type,
            x1: Math.min(editorStartPos.x, end.x),
            y1: Math.min(editorStartPos.y, end.y),
            x2: Math.max(editorStartPos.x, end.x),
            y2: Math.max(editorStartPos.y, end.y)
        };
        state[state.editorTarget].boxes.push(box);
        renderEditor();
        renderBoxList(state.editorTarget);
    }
    editorStartPos = null;
});

els.closeEditor.addEventListener('click', closeEditor);
els.undoBoxBtn.addEventListener('click', () => {
    if (state.editorTarget) {
        state[state.editorTarget].boxes.pop();
        renderEditor();
        renderBoxList(state.editorTarget);
    }
});
els.clearBoxesBtn.addEventListener('click', () => {
    if (state.editorTarget) {
        state[state.editorTarget].boxes = [];
        renderEditor();
        renderBoxList(state.editorTarget);
    }
});
els.editorModal.addEventListener('click', (e) => {
    if (e.target === els.editorModal) closeEditor();
});

// ===================== 置信度与显示选项 =====================
els.confidence.addEventListener('input', () => {
    els.confidenceValue.textContent = parseFloat(els.confidence.value).toFixed(2);
});

[els.showBoxes, els.showMasks, els.showLabels].forEach(el => {
    el.addEventListener('change', () => {
        if (currentFile && lastResults.length > 0) renderResult(currentFile, lastResults);
    });
});

els.displayConfidence.addEventListener('input', () => {
    displayConfidenceThreshold = parseFloat(els.displayConfidence.value);
    els.displayConfidenceValue.textContent = displayConfidenceThreshold.toFixed(2);
    if (currentFile && lastResults.length > 0) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

els.minAreaInput.addEventListener('input', () => {
    const value = parseInt(els.minAreaInput.value, 10);
    minAreaThreshold = isNaN(value) || value < 0 ? 0 : value;
    if (currentFile && lastResults.length > 0) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

els.sortBy.addEventListener('change', () => {
    sortMode = els.sortBy.value;
    if (currentFile && lastResults.length > 0) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

els.toggleShowChecked.addEventListener('click', () => {
    showOnlyChecked = !showOnlyChecked;
    els.toggleShowChecked.textContent = showOnlyChecked ? '展示全部结果' : '只展示已勾选';
    els.toggleShowChecked.classList.toggle('secondary', !showOnlyChecked);
    els.toggleShowChecked.classList.toggle('active', showOnlyChecked);
    if (currentFile && lastResults.length > 0) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

els.clearCheckedBtn.addEventListener('click', () => {
    checkedResultIndices.clear();
    if (showOnlyChecked) {
        showOnlyChecked = false;
        els.toggleShowChecked.textContent = '只展示已勾选';
        els.toggleShowChecked.classList.add('secondary');
        els.toggleShowChecked.classList.remove('active');
    }
    if (currentFile && lastResults.length > 0) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

// ===================== Omnicrop 配置折叠 =====================
els.cropConfigToggle.addEventListener('click', () => {
    const hidden = els.cropConfigPanel.style.display === 'none';
    els.cropConfigPanel.style.display = hidden ? 'block' : 'none';
    els.cropConfigToggle.lastElementChild.textContent = hidden ? '▲' : '▼';
});

function getCropConfig() {
    return {
        max_size: parseInt(els.cropMaxSize.value) || 640,
        padding: parseInt(els.cropPadding.value) || 20,
        w_diou: parseFloat(els.cropWDiou.value) || 30.0,
        w_expansion: parseFloat(els.cropWExpansion.value) || 5.0,
        count_penalty: parseFloat(els.cropCountPenalty.value) || 120.0,
        nms_threshold: parseFloat(els.cropNmsThreshold.value) || 0.2,
        enable_ar_fix: els.cropEnableAr.checked,
        target_ar: parseFloat(els.cropTargetAr.value) || 1.0
    };
}

// ===================== 提交推理 =====================
els.submitBtn.addEventListener('click', async () => {
    if (!state.target.file) {
        showToast('请先选择目标图片', 'error');
        return;
    }

    const mode = state.mode;
    const needTags = ['multi-class', 'mixed', 'obj-refine'].includes(mode);
    if (needTags && tags.length === 0) {
        showToast('请至少保留一个识别标签', 'error');
        return;
    }

    if ((mode === 'box' || mode === 'mixed') && state.target.boxes.length === 0) {
        showToast('请至少标注一个目标 Box', 'error');
        return;
    }

    if (mode === 'from-image') {
        if (!state.prompt.file) {
            showToast('请上传参考图片', 'error');
            return;
        }
        if (state.prompt.boxes.length === 0) {
            showToast('请在参考图上至少标注一个 Box', 'error');
            return;
        }
    }

    els.submitBtn.disabled = true;
    els.submitBtn.textContent = '检测中...';
    els.stats.textContent = '';

    const formData = new FormData();
    formData.append('mode', mode);
    formData.append('image', state.target.file);
    formData.append('confidence', els.confidence.value);
    formData.append('return_mask', els.returnMask.checked);

    if (needTags) {
        tags.forEach(name => formData.append('class_names', name));
    }

    if (mode === 'box' || mode === 'mixed') {
        formData.append('target_boxes', JSON.stringify(state.target.boxes));
    }

    if (mode === 'from-image') {
        formData.append('prompt_image', state.prompt.file);
        formData.append('prompt_boxes', JSON.stringify(state.prompt.boxes));
    }

    if (mode === 'obj-refine') {
        formData.append('pre_detect_labels', predefinedTags.join(','));
        formData.append('merge_results', els.mergeResults.checked);
        formData.append('crop_config_json', JSON.stringify(getCropConfig()));
    }

    try {
        const response = await fetch('/predict/file', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || '请求失败');
        }

        els.jsonOutput.value = JSON.stringify(data, null, 2);
        lastResults = data.results || [];
        currentFile = state.target.file;
        renderFilterPanel(lastResults);
        await renderResult(currentFile, lastResults);

        els.placeholder.style.display = 'none';
        els.resultContainer.style.display = 'block';
        updateStats();
        showToast('检测完成');
    } catch (err) {
        showToast('检测失败：' + err.message, 'error');
    } finally {
        els.submitBtn.disabled = false;
        els.submitBtn.textContent = '开始检测';
    }
});

// ===================== 结果渲染 =====================
function sortVisibleItems(items) {
    if (sortMode === 'default') return items;
    const sorted = [...items];
    switch (sortMode) {
        case 'area_desc': sorted.sort((a, b) => getBoxArea(b.item) - getBoxArea(a.item)); break;
        case 'area_asc': sorted.sort((a, b) => getBoxArea(a.item) - getBoxArea(b.item)); break;
        case 'score_desc': sorted.sort((a, b) => b.item.score - a.item.score); break;
        case 'score_asc': sorted.sort((a, b) => a.item.score - b.item.score); break;
    }
    return sorted;
}

function updateChipVisual(chip, label, isActive) {
    const color = getColor(label);
    const rgb = hexToRgb(color);
    if (isActive) {
        chip.classList.remove('inactive');
        chip.style.borderColor = color;
        chip.style.background = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.12)`;
        chip.style.color = color;
    } else {
        chip.classList.add('inactive');
        chip.style.borderColor = '';
        chip.style.background = '';
        chip.style.color = '';
    }
}

function renderFilterPanel(results) {
    const labels = [...new Set(results.map(r => r.label))];
    activeLabels = new Set(labels);
    displayConfidenceThreshold = 0.0;
    els.displayConfidence.value = '0.0';
    els.displayConfidenceValue.textContent = '0.00';
    minAreaThreshold = 0;
    els.minAreaInput.value = '0';

    if (labels.length === 0) {
        els.filterPanel.style.display = 'none';
    } else {
        els.filterPanel.style.display = 'block';
        els.filterList.innerHTML = '';
        els.filterList.className = 'filter-chip-list';
        labels.forEach((label) => {
            const color = getColor(label);
            const chip = document.createElement('div');
            chip.className = 'filter-chip';
            chip.dataset.label = label;
            chip.innerHTML = `
                <span class="chip-dot" style="background:${color}"></span>
                <span>${label}</span>
                <span class="chip-check" style="background:${color};color:#fff">✓</span>
            `;
            updateChipVisual(chip, label, true);
            chip.addEventListener('click', () => {
                if (activeLabels.has(label)) activeLabels.delete(label);
                else activeLabels.add(label);
                updateChipVisual(chip, label, activeLabels.has(label));
                if (currentFile) {
                    renderResult(currentFile, lastResults);
                    updateStats();
                }
            });
            els.filterList.appendChild(chip);
        });
    }

    els.scoreFilterPanel.style.display = results.length > 0 ? 'block' : 'none';
    els.areaFilterPanel.style.display = results.length > 0 ? 'block' : 'none';
}

els.selectAllBtn.addEventListener('click', () => {
    els.filterList.querySelectorAll('.filter-chip').forEach(chip => {
        const label = chip.dataset.label;
        activeLabels.add(label);
        updateChipVisual(chip, label, true);
    });
    if (currentFile) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

els.selectNoneBtn.addEventListener('click', () => {
    els.filterList.querySelectorAll('.filter-chip').forEach(chip => {
        const label = chip.dataset.label;
        activeLabels.delete(label);
        updateChipVisual(chip, label, false);
    });
    if (currentFile) {
        renderResult(currentFile, lastResults);
        updateStats();
    }
});

function rleDecodeToCanvas(rle, width, height, color = null) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;
    const rgb = color ? hexToRgb(color) : { r: 255, g: 255, b: 255 };
    for (let i = 0; i < rle.length; i += 2) {
        const start = rle[i] - 1;
        const len = rle[i + 1];
        for (let j = 0; j < len; j++) {
            const idx = start + j;
            if (idx >= 0 && idx < width * height) {
                const p = idx * 4;
                data[p] = rgb.r;
                data[p + 1] = rgb.g;
                data[p + 2] = rgb.b;
                data[p + 3] = 255;
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas;
}

function renderDetectionList(visibleItems) {
    els.resultListPanel.style.display = 'flex';
    els.detectionList.style.display = 'block';
    const hasChecked = checkedResultIndices.size > 0;
    els.clearCheckedBtn.style.display = hasChecked ? 'block' : 'none';

    els.detectionList.innerHTML = '';
    if (visibleItems.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        if (showOnlyChecked && checkedResultIndices.size === 0) {
            empty.textContent = '当前没有勾选任何结果，请先在列表中勾选目标，或切换为“展示全部结果”。';
        } else if (showOnlyChecked) {
            empty.textContent = '已开启“只展示已勾选”，但当前筛选条件无匹配结果。';
        } else {
            empty.textContent = '当前筛选条件无匹配结果。';
        }
        els.detectionList.appendChild(empty);
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'detection-grid';
    visibleItems.forEach(({ item, originalIndex }, idx) => {
        const area = getBoxArea(item);
        const boxStr = item.box ? item.box.map(v => Math.round(v)).join(', ') : '-';
        const isChecked = checkedResultIndices.has(originalIndex);
        const card = document.createElement('div');
        card.className = 'detection-card' +
            (originalIndex === selectedResultIndex ? ' selected' : '') +
            (isChecked ? ' checked' : '');
        card.innerHTML = `
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
                <input type="checkbox" class="result-check" data-idx="${originalIndex}" ${isChecked ? 'checked' : ''}>
                <span style="font-weight:600;line-height:1.3;flex:1">
                    <span style="display:inline-block;width:8px;height:8px;background:${getColor(item.label)};margin-right:4px;border-radius:2px"></span>
                    #${idx + 1} ${item.label}
                </span>
            </div>
            <div style="line-height:1.4;padding-left:22px">${(item.score * 100).toFixed(0)}% · ${area.toLocaleString()}px²</div>
            <div style="color:#64748b;font-size:10px;line-height:1.3;margin-top:1px;padding-left:22px">[${boxStr}]</div>
        `;
        const checkbox = card.querySelector('.result-check');
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
            if (checkbox.checked) checkedResultIndices.add(originalIndex);
            else checkedResultIndices.delete(originalIndex);
            renderResult(currentFile, lastResults);
        });
        card.addEventListener('click', async (e) => {
            if (e.target.classList.contains('result-check')) return;
            selectedResultIndex = originalIndex;
            await renderResult(currentFile, lastResults);
            const selectedCard = els.detectionList.querySelector('.detection-card.selected');
            if (selectedCard) selectedCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
        grid.appendChild(card);
    });
    els.detectionList.appendChild(grid);
}

function renderResult(file, results) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            els.sourceImage.src = e.target.result;
            els.sourceImage.onload = () => {
                const width = els.sourceImage.naturalWidth;
                const height = els.sourceImage.naturalHeight;
                els.overlayCanvas.width = width;
                els.overlayCanvas.height = height;
                const ctx = els.overlayCanvas.getContext('2d');
                ctx.clearRect(0, 0, width, height);

                const composeCanvas = document.createElement('canvas');
                composeCanvas.width = width;
                composeCanvas.height = height;
                const cctx = composeCanvas.getContext('2d');
                cctx.drawImage(els.sourceImage, 0, 0);

                els.legend.innerHTML = '';
                els.detectionList.innerHTML = '';
                els.detectionList.style.display = 'none';
                const labelSet = new Set();

                const doBoxes = els.showBoxes.checked;
                const doMasks = els.showMasks.checked;
                const doLabels = els.showLabels.checked;

                lastVisibleItems = [];
                results.forEach((item, originalIndex) => {
                    if (!isResultVisible(item)) return;
                    if (showOnlyChecked && !checkedResultIndices.has(originalIndex)) return;
                    lastVisibleItems.push({ item, originalIndex });
                });
                lastVisibleItems = sortVisibleItems(lastVisibleItems);

                function drawItem({ item, originalIndex }, isHighlight, isChecked) {
                    const [x1, y1, x2, y2] = item.box;
                    const label = item.label;
                    const color = getColor(label);
                    const score = (item.score * 100).toFixed(1) + '%';
                    const alpha = isHighlight ? 0.65 : 0.45;
                    const lineWidth = isHighlight ? Math.max(4, width / 250) : Math.max(2, width / 400);

                    if (doMasks && item.mask) {
                        const mw = item.mask_width || Math.max(1, x2 - x1);
                        const mh = item.mask_height || Math.max(1, y2 - y1);
                        const maskCanvas = rleDecodeToCanvas(item.mask, mw, mh, color);
                        const bw = x2 - x1;
                        const bh = y2 - y1;

                        ctx.save();
                        ctx.globalAlpha = alpha;
                        ctx.drawImage(maskCanvas, x1, y1, bw, bh);
                        ctx.restore();

                        cctx.save();
                        cctx.globalAlpha = alpha;
                        cctx.drawImage(maskCanvas, x1, y1, bw, bh);
                        cctx.restore();
                    }

                    if (doBoxes) {
                        ctx.strokeStyle = color;
                        ctx.lineWidth = lineWidth;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                        cctx.strokeStyle = color;
                        cctx.lineWidth = lineWidth;
                        cctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                        if (isChecked) {
                            const checkColor = '#facc15';
                            ctx.strokeStyle = checkColor;
                            ctx.lineWidth = Math.max(3, width / 300);
                            ctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);

                            cctx.strokeStyle = checkColor;
                            cctx.lineWidth = Math.max(3, width / 300);
                            cctx.strokeRect(x1 - 2, y1 - 2, x2 - x1 + 4, y2 - y1 + 4);
                        }

                        if (isHighlight) {
                            ctx.strokeStyle = '#fff';
                            ctx.lineWidth = Math.max(2, width / 500);
                            ctx.strokeRect(x1 - 3 - (isChecked ? 1 : 0), y1 - 3 - (isChecked ? 1 : 0), x2 - x1 + 6 + (isChecked ? 2 : 0), y2 - y1 + 6 + (isChecked ? 2 : 0));
                        }
                    }

                    if (doLabels && doBoxes) {
                        const text = `${label} ${score}`;
                        ctx.font = `bold ${Math.max(12, width / 60)}px sans-serif`;
                        const textMetrics = ctx.measureText(text);
                        const textHeight = Math.max(16, width / 50);
                        ctx.fillStyle = color;
                        ctx.fillRect(x1, y1 - textHeight, textMetrics.width + 10, textHeight);
                        ctx.fillStyle = '#fff';
                        ctx.fillText(text, x1 + 5, y1 - 4);

                        cctx.font = ctx.font;
                        cctx.fillStyle = color;
                        cctx.fillRect(x1, y1 - textHeight, textMetrics.width + 10, textHeight);
                        cctx.fillStyle = '#fff';
                        cctx.fillText(text, x1 + 5, y1 - 4);
                    }

                    if (!labelSet.has(label)) {
                        labelSet.add(label);
                        const div = document.createElement('div');
                        div.className = 'legend-item';
                        div.innerHTML = `<span class="legend-color" style="background:${color}"></span>${label}`;
                        els.legend.appendChild(div);
                    }
                }

                lastVisibleItems.forEach(item => drawItem(item, false, checkedResultIndices.has(item.originalIndex)));

                if (selectedResultIndex !== null) {
                    const selected = lastVisibleItems.find(v => v.originalIndex === selectedResultIndex);
                    if (selected) drawItem(selected, true, checkedResultIndices.has(selected.originalIndex));
                }

                renderDetectionList(lastVisibleItems);

                lastResultImage = composeCanvas;
                resolve();
            };
        };
        reader.readAsDataURL(file);
    });
}

els.downloadBtn.addEventListener('click', () => {
    if (!lastResultImage) return;
    const link = document.createElement('a');
    link.download = 'result_' + Date.now() + '.png';
    link.href = lastResultImage.toDataURL('image/png');
    link.click();
});

els.overlayCanvas.addEventListener('click', (e) => {
    if (!currentFile || lastVisibleItems.length === 0) return;
    const rect = els.overlayCanvas.getBoundingClientRect();
    const scaleX = els.overlayCanvas.width / rect.width;
    const scaleY = els.overlayCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    for (let i = lastVisibleItems.length - 1; i >= 0; i--) {
        const { item, originalIndex } = lastVisibleItems[i];
        const [x1, y1, x2, y2] = item.box;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
            selectedResultIndex = originalIndex;
            renderResult(currentFile, lastResults);
            break;
        }
    }
});

// ===================== 初始化 =====================
renderTags();
renderPredefinedTags();
updateModeUI();
