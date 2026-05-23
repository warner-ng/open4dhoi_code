$(document).ready(function() {
    let totalFrames = 0;
    let currentFrame = 0;
    let isPlaying = false;
    let fps = 30; // Default fps, will be updated from video metadata
    let playInterval = null;
    let playAnimationFrameId = null; // For requestAnimationFrame
    let sliderUpdateTimeout = null; // For throttling slider updates
    let dragDebounceTimer = null; // For debouncing frame updates during dragging
    let meshData = null;
    let meshVertexColors = null; // Vertex color array (if the .obj file contains colors)
    let selectedPoints = []; // Object points
    let activeObjectPointIndex = -1; // The currently selected object point for annotation
    let humanKeypoints = {}; // { jointName: {index, x, y, z} }
    // New structure for annotations:
    // annotations[objPointIndex] = { type: 'human', joint: 'name' } OR { type: '2d', trackId: '...' }
    // But to keep it simple and compatible with existing save logic:
    // humanKeypoints maps Joint -> ObjPoint.
    // We need a map ObjPoint -> Joint to easily check status.
    // And ObjPoint -> 2DTrack.
    let objPointToJoint = {}; // { objIdx: jointName } for current frame only
    let objPointToTrack = {}; // { objIdx: { frameIdx: [x, y] } }

    // For 3D human-joint mapping, we want edits at a given
    // frame to only affect that frame and later frames, not
    // earlier frames. We therefore maintain per-object keyframes
    // over time and derive objPointToJoint for the current frame
    // from those keyframes.
    // jointKeyframesByObj[objIdx] = [ { frame: int, joint: string|null }, ... ]
    let jointKeyframesByObj = {};

    // Similarly, for 3D point visibility on the object surface,
    // we keep per-object visibility keyframes so that deleting
    // a point at a given frame hides it from that frame onward
    // without affecting earlier frames.
    // visibilityKeyframesByObj[objIdx] = [ { frame: int, visible: bool }, ... ]
    let visibilityKeyframesByObj = {};
    
    let currentMode = 'view'; // 'view', 'select', 'delete'
    let selectedJointName = null;
    let jointTree = null;
    let mainJointCoords = null;
    let buttonNames = null;

    // Scale-check state
    let hasCheckedScale = false; // forces user to open the Check Scale view before annotating
    let scaleViewerDocControlsBound = false;
    let lastAppliedScale = 1.0;
    const scaleViewerState = {
        baseHuman: null,
        baseObject: null,
        baseScaleFactor: 1.0,
        layout: null,
        baseDiag: 1,
        transform: { tx: 0, ty: 0, tz: 0, yaw: 0, pitch: 0, roll: 0, viewRoll: 0 },
        activeMode: null,
        lastMouse: null,
        camera: {
            eye: { x: 1.25, y: 1.25, z: 1.25 },
            up: { x: 0, y: 0, z: 1 },
            center: { x: 0, y: 0, z: 0 }
        }
    };

    // Frame cache-busting key to avoid stale frames across sessions
    let frameCacheKey = Date.now();

    // 2D Annotation variables
    let pending2DPoint = null; // last clicked 2D point (for legacy use / highlighting)
    let pending2DPoints = {};  // { objIdx: { x, y, displayX, displayY, frame } }

        // Frame preloading cache
        let preloadCache = new Map();

        // Preload adjacent frames (optimized with cache)
        function preloadFrames(frameNum) {
            const preloadCount = 2; // Reduce preload count to save memory
            const framesToPreload = [];

            // Collect frames to preload
            for (let i = 1; i <= preloadCount; i++) {
                if (frameNum + i < totalFrames && !preloadCache.has(frameNum + i)) {
                    framesToPreload.push(frameNum + i);
                }
                if (frameNum - i >= 0 && !preloadCache.has(frameNum - i)) {
                    framesToPreload.push(frameNum - i);
                }
            }

            // Limit cache size to avoid memory leaks
            if (preloadCache.size > 50) {
                // Clear cache entries farthest from current frame
                const sortedKeys = Array.from(preloadCache.keys()).sort((a, b) =>
                    Math.abs(a - frameNum) - Math.abs(b - frameNum)
                );
                // Keep the nearest 30, remove the rest
                sortedKeys.slice(30).forEach(key => {
                    preloadCache.delete(key);
                });
            }

            // Preload new frames
            framesToPreload.forEach(frameNum => {
                const img = new Image();
                img.onload = () => {
                    // Add to cache after image finishes loading
                    preloadCache.set(frameNum, img);
                };
                img.src = 'api/frame/' + frameNum + '?v=' + frameCacheKey;
            });
        }
    
    // Plotly variables
    let meshTrace = null;
    let scatterTrace = null; // Object points (Red)
    let humanTrace = null;   // Human keypoints (Green)
    let layout = null;

    // Currently selected session_folder (from upload_records.json)
    let currentSessionFolder = null;
    let selectedSessionFolder = null;
    let currentUser = null;
    let annotatorFromConfig = null;  // Annotator configured from config.yaml

    function setUserUI(user, fromConfig) {
        currentUser = user;
        if (fromConfig) {
            // Annotator from config file, do not show login/logout buttons
            $('#user-name').text(user.username + ' (config)');
            $('#btn-login').hide();
            $('#btn-logout').hide();
        } else if (user && user.username) {
            $('#user-name').text(user.display_name || user.username);
            $('#btn-login').hide();
            $('#btn-logout').show();
        } else {
            $('#user-name').text('Not logged in');
            $('#btn-login').show();
            $('#btn-logout').hide();
        }
    }

    function fetchAnnotator() {
        // First check if an annotator is configured in config.yaml
        return $.getJSON('api/annotator').done(function(resp) {
            if (resp.ok && resp.annotator && resp.source === 'config') {
                annotatorFromConfig = resp.annotator;
                setUserUI({ username: resp.annotator, display_name: resp.annotator }, true);
            } else {
                // No annotator configured, fall back to login system
                fetchMe();
            }
        }).fail(function() {
            fetchMe();
        });
    }

    function fetchMe() {
        return $.getJSON('api/me').done(function(resp) {
            if (resp.ok && resp.logged_in) {
                setUserUI(resp.user, false);
            } else {
                setUserUI(null, false);
            }
        }).fail(function() {
            setUserUI(null, false);
        });
    }

    function promptLogin(cb) {
        // If an annotator is already configured, return success directly
        if (annotatorFromConfig) {
            if (cb) cb(true);
            return;
        }
        const name = window.prompt('Please enter username to login');
        if (!name) {
            if (cb) cb(false);
            return;
        }
        $.ajax({
            url: 'api/login',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ username: name.trim() })
        }).done(function(resp) {
            setUserUI({ username: resp.username, display_name: resp.username }, false);
            if (cb) cb(true);
        }).fail(function(xhr) {
            const msg = (xhr.responseJSON && xhr.responseJSON.error) || xhr.statusText;
            alert('Login failed: ' + msg);
            if (cb) cb(false);
        });
    }

    $('#btn-login').on('click', function() {
        promptLogin(function(ok) {
            if (ok) {
                loadHoiTasks();
            }
        });
    });

    $('#btn-logout').on('click', function() {
        $.post('api/logout').always(function() {
            setUserUI(null);
            currentSessionFolder = null;
            selectedSessionFolder = null;
            $('#hoi-status').text('Logged out');
            loadHoiTasks();
        });
    });

    // Initialize
    fetchAnnotator().always(function() {
        loadJointTree();
        loadHumanSelectorData();
        // Load HOI list first; after user selects and clicks "Start Annotation", fetch metadata / mesh
        loadHoiTasks();
    });

    // Optimization Button Handler (optimize from start frame to end of sequence)
    $('#btn-optimize').click(function() {
        const btn = $(this);
        if (btn.data('running')) {
            return; // Prevent double-clicks while a run is active
        }

        const originalText = btn.text();
        const restoreButton = () => {
            btn.data('running', false);
            btn.prop('disabled', false);
            btn.text(originalText);
        };

        btn.data('running', true);
        btn.prop('disabled', true);
        btn.text('Optimizing...');

        try {
            // 1. Save merged annotations first (no progress update here)
            const saveReq = saveMergedAnnotations(null, { update_progress: false });

            if (!saveReq || typeof saveReq.done !== 'function') {
                alert('Failed to start save request before optimization.');
                restoreButton();
                return;
            }

            saveReq
                .done(function() {
                    try {
                        // 2. Run optimization to the end of sequence by default
                        const safeLastFrame = (typeof totalFrames !== 'undefined' && Number.isFinite(totalFrames) && totalFrames > 0)
                            ? (totalFrames - 1)
                            : currentFrame;
                        const optReq = $.ajax({
                            url: 'api/run_optimization',
                            type: 'POST',
                            contentType: 'application/json',
                            data: JSON.stringify({
                                frame_idx: currentFrame,
                                last_frame: safeLastFrame
                            })
                        });

                        optReq
                            .done(function(response) {
                                if (response.status === 'success') {
                                    alert('Optimization completed successfully!');
                                    try {
                                        updateSceneData(currentFrame);
                                    } catch (e) {
                                        console.warn('updateSceneData failed:', e);
                                    }
                                } else {
                                    alert('Optimization failed: ' + (response.message || 'Unknown error'));
                                }
                            })
                            .fail(function(xhr, status, error) {
                                console.error('Optimization error:', error);
                                const msg = (xhr.responseJSON && xhr.responseJSON.message) || error || status;
                                alert('Error running optimization: ' + msg);
                            })
                            .always(restoreButton);
                    } catch (e) {
                        console.error('Failed to start optimization request:', e);
                        alert('Error starting optimization: ' + (e && e.message ? e.message : String(e)));
                        restoreButton();
                    }
                })
                .fail(function(xhr, status, error) {
                    const msg = (xhr && xhr.responseJSON && xhr.responseJSON.error) || error || status || 'Unknown error';
                    alert('Failed to save annotations before optimization: ' + msg);
                    restoreButton();
                });
        } catch (e) {
            console.error('Optimize click handler failed:', e);
            alert('Optimize failed: ' + (e && e.message ? e.message : String(e)));
            restoreButton();
        }
    });
    

    // Load existing annotations if any (mock for now, or fetch from server)
    // In a real app, we would fetch saved annotations here.
    // For now, we start fresh or rely on what's in memory if page isn't reloaded.

    function loadJointTree() {
        $.getJSON('asset/data/joint_tree.json', function(data) {
            jointTree = data;
            initJointDropdown();
        });
    }

    function loadHumanSelectorData() {
        $.when(
            $.getJSON('asset/data/main_joint.json'),
            $.getJSON('asset/data/button_name.json')
        ).done(function(coords, names) {
            mainJointCoords = coords[0];
            buttonNames = names[0];
            initHumanKeypointSelector();
        });
    }

    function initHumanKeypointSelector() {
        const container = $('#human-image-container');
        const img = $('#human-ref-img');
        
        // Wait for image to load to get dimensions
        if (img[0].complete) {
            renderButtons();
        } else {
            img.on('load', renderButtons);
        }

        function renderButtons() {
            const naturalWidth = img[0].naturalWidth;
            const naturalHeight = img[0].naturalHeight;

            // Fine-tuning offsets (in ORIGINAL IMAGE pixels).
            // If buttons appear shifted (e.g., down-left), adjust these small values.
            // Typical ranges: -30..30
            const HUMAN_BTN_OFFSET_X = 0;
            const HUMAN_BTN_OFFSET_Y = 0;
            
            // Use a fixed reference size for scaling logic to match original coordinates
            // The original app likely used 480x480 as the display size for the coordinates.
            const referenceSize = 480;
            
            // Get current container dimensions
            const containerWidth = container.width();
            
            // Calculate scale factor based on how much the container has shrunk/grown relative to reference
            // Assuming the image fills the container and maintains aspect ratio (square)
            const scaleFactor = containerWidth / referenceSize;
            
            // Clear existing buttons
            container.find('.joint-btn').remove();

            for (const [jointName, coords] of Object.entries(mainJointCoords)) {
                let [realX, realY] = coords;

                // Apply small global offsets in the original image coordinate space.
                realX = realX + HUMAN_BTN_OFFSET_X;
                realY = realY + HUMAN_BTN_OFFSET_Y;

                // Calculate position as percentage of container
                // This makes it responsive to any size
                let percentX = (realX / naturalWidth) * 100;
                let percentY = (realY / naturalHeight) * 100;
                
                const btnLabel = buttonNames[jointName] || jointName;
                
                const btn = $('<div class="joint-btn"></div>')
                    .text(btnLabel)
                    .css({
                        left: percentX + '%',
                        top: percentY + '%',
                        fontSize: Math.max(10, 10 * scaleFactor) + 'px', // Scale font but min 10px
                        padding: Math.max(2, 2 * scaleFactor) + 'px',
                        position: 'absolute',
                        transform: 'translate(-50%, -50%)' // Center button on point
                    })
                    .attr('title', jointName);
                
                btn.click(function(e) {
                    e.stopPropagation();
                    showContextMenu(jointName, e.pageX, e.pageY);
                });

                container.append(btn);
            }
        }
        
        // Re-render on resize
        $(window).resize(function() {
             if (container.is(':visible')) renderButtons();
        });
        
        // Also re-render when tab is switched
        $('#tab-human').click(function() {
             setTimeout(renderButtons, 50);
        });
    }

    function showContextMenu(mainJoint, x, y) {
        // Remove existing context menu
        $('.context-menu').remove();

        const menu = $('<div class="context-menu"></div>');

        // Add sub-joints
        const subJoints = jointTree[mainJoint] || [];

        // Add main joint itself as an option
        const mainItem = $('<div class="context-menu-item"></div>')
            .text(mainJoint + " (Main)")
            .click(function() {
                selectJoint(mainJoint);
                menu.remove();
            });
        menu.append(mainItem);

        if (subJoints.length > 0) {
            menu.append('<div class="context-menu-separator"></div>');
            subJoints.forEach(sub => {
                const item = $('<div class="context-menu-item"></div>')
                    .text(sub)
                    .click(function() {
                        selectJoint(sub);
                        menu.remove();
                    });
                menu.append(item);
            });
        }

        // Add Cancel
        menu.append('<div class="context-menu-separator"></div>');
        const cancelItem = $('<div class="context-menu-item"></div>')
            .text("Cancel")
            .click(function() {
                menu.remove();
            });
        menu.append(cancelItem);

        $('body').append(menu);

        // Position menu - for leftFoot and rightFoot, show menu above
        const menuHeight = menu.outerHeight();
        let posY = y;
        if (mainJoint === 'leftFoot' || mainJoint === 'rightFoot') {
            // Show above the click position
            posY = y - menuHeight - 10;
            if (posY < 0) posY = 10; // Ensure it does not go beyond the top of the screen
        }

        menu.css({
            left: x + 'px',
            top: posY + 'px',
            display: 'block'
        });

        // Close on click outside
        $(document).one('click', function() {
            menu.remove();
        });
    }

    function initJointDropdown() {
        // Deprecated dropdown logic, but we can use it to populate a tree view if needed.
        // For now, the visual selector is primary.
    }

    function selectJoint(name) {
        selectedJointName = name;
        $('#selected-joint-display').text(name);
        // If we have an active object point, automatically link it?
        // User might want to confirm. But previous logic was auto-click if active.
        // Let's keep it manual for now as per "Link to Joint" button presence.
        // Actually, user said "click human joint OR click 2D point".
        // So maybe auto-link is better?
        // "Link to Joint" button is there. Let's use it.
    }

    // ===== 3D Joint Mapping over Time (per-frame semantics) =====

    function addJointKeyframe(objIdx, frame, jointName) {
        if (!jointKeyframesByObj[objIdx]) {
            jointKeyframesByObj[objIdx] = [];
        }
        jointKeyframesByObj[objIdx].push({ frame: frame, joint: jointName });
        // Keep keyframes sorted by frame
        jointKeyframesByObj[objIdx].sort((a, b) => a.frame - b.frame);
    }

    function getJointForObjectAtFrame(objIdx, frame) {
        const kfs = jointKeyframesByObj[objIdx];
        if (!kfs || kfs.length === 0) return null;
        let result = null;
        for (const kf of kfs) {
            if (kf.frame <= frame) {
                result = kf.joint;
            } else {
                break;
            }
        }
        return result;
    }

    function applyJointMappingForCurrentFrame() {
        // Recompute objPointToJoint for the current frame from keyframes
        objPointToJoint = {};
        selectedPoints.forEach(pt => {
            const j = getJointForObjectAtFrame(pt.index, currentFrame);
            if (j) {
                objPointToJoint[pt.index] = j;
            }
        });
    }

    function addVisibilityKeyframe(objIdx, frame, visible) {
        if (!visibilityKeyframesByObj[objIdx]) {
            visibilityKeyframesByObj[objIdx] = [];
        }
        visibilityKeyframesByObj[objIdx].push({ frame: frame, visible: !!visible });
        visibilityKeyframesByObj[objIdx].sort((a, b) => a.frame - b.frame);
    }

    function isObjectVisibleAtFrame(objIdx, frame) {
        const kfs = visibilityKeyframesByObj[objIdx];
        if (!kfs || kfs.length === 0) return true; // default visible
        let result = true;
        for (const kf of kfs) {
            if (kf.frame <= frame) {
                result = kf.visible;
            } else {
                break;
            }
        }
        return result;
    }

    // Tab Switching Logic
    $('#tab-human').click(function() {
        $('.panel-tab').removeClass('active').css({
            'background': 'transparent',
            'color': '#666'
        });
        $(this).addClass('active').css({
            'background': 'white',
            'color': '#667eea'
        });
        $('#panel-human-joints').show();
        $('#panel-2d-view').hide();
    });

    $('#tab-2d').click(function() {
        $('.panel-tab').removeClass('active').css({
            'background': 'transparent',
            'color': '#666'
        });
        $(this).addClass('active').css({
            'background': 'white',
            'color': '#667eea'
        });
        $('#panel-human-joints').hide();
        $('#panel-2d-view').css('display', 'flex');
        init2DCanvas();
    });

    // Event Listeners
    $('#play-pause').click(togglePlay);
    
    // Use 'input' event for immediate response during dragging
    // Don't pause playback, just update frame immediately
    let isDragging = false;
    let lastUpdateTime = 0;
    const minUpdateInterval = 16; // ~60fps max update rate
    
    $('#frame-slider').on('mousedown', function() {
        isDragging = true;
        // Don't pause playback - let it continue, but slider will override
    });
    
    $('#frame-slider').on('input', function() {
        const newFrame = parseInt($(this).val());
        if (newFrame !== currentFrame) {
            // Update progress display immediately for responsive UI
            updateProgressDisplay(newFrame);

            // Clear previous debounce timer
            if (dragDebounceTimer) {
                clearTimeout(dragDebounceTimer);
            }

            // Set new debounce timer (100ms delay)
            dragDebounceTimer = setTimeout(function() {
                loadFrame(newFrame);
                dragDebounceTimer = null;
            }, 100);
        }
    });
    
    $('#frame-slider').on('mouseup', function() {
        isDragging = false;
        // Cancel any pending debounce timer and update immediately
        if (dragDebounceTimer) {
            clearTimeout(dragDebounceTimer);
            dragDebounceTimer = null;
        }
        currentFrame = parseInt($(this).val());
        loadFrame(currentFrame);
    });
    
    // Also handle change event for final update (touch devices)
    $('#frame-slider').on('change', function() {
        currentFrame = parseInt($(this).val());
        loadFrame(currentFrame);
    });
    // Save buttons: per-frame inside annotation modal, and global final-save button on main page
    $('#save-annotation').click(saveAnnotation);
    $('#save-annotation-main').click(saveAllAnnotations);

    // Mode Switching
    $('.mode-btn').click(function() {
        // Ignore special buttons
        if (['btn-set-joint', 'btn-show-selector', 'btn-track-2d', 'btn-manage', 'btn-toggle-2d'].includes(this.id)) return;
        
        $('.mode-btn').removeClass('active');
        $(this).addClass('active');
        
        if (this.id === 'mode-view') currentMode = 'view';
        else if (this.id === 'mode-select') currentMode = 'select';
        else if (this.id === 'mode-delete') currentMode = 'delete';
        
        // Update Plotly dragmode
        const dragmode = currentMode === 'view' ? 'orbit' : 'orbit'; // Always orbit, but click behavior changes
        Plotly.relayout('3d-viewer', { 'scene.dragmode': dragmode });
        
        console.log("Switched to mode:", currentMode);
    });

    // Removed 2D View Toggle logic as it is now tab-based

    function init2DCanvas() {
        const img = document.getElementById('modal-video-frame');
        const canvas = document.getElementById('modal-video-overlay');
        
        // Wait for image load if needed
        if (img.complete) {
            resizeCanvas();
        } else {
            img.onload = resizeCanvas;
        }
        
        function resizeCanvas() {
            // Set canvas size to match the image element's rendered size
            // Note: If object-fit: contain is used, clientWidth/Height includes black bars.
            // We should ideally match the actual image area, but for simplicity we match the element
            // and handle coordinate mapping in the click handler.
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
        }
        
        // Handle resize
        $(window).resize(resizeCanvas);
    }

    function runTrackingForObject(objIdx, startFrame, x, y, onDone, onError) {
        $.ajax({
            url: 'api/track_2d',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                frame_idx: startFrame,
                x: x,
                y: y
            }),
            success: function(response) {
                const filteredTracks = {};
                for (const [f, pt] of Object.entries(response.tracks || {})) {
                    const fi = parseInt(f);
                    if (!Number.isNaN(fi) && fi >= startFrame) {
                        filteredTracks[fi] = pt;
                    }
                }

                const existing = objPointToTrack[objIdx] || {};
                const merged = { ...existing };
                for (const [fi, pt] of Object.entries(filteredTracks)) {
                    merged[fi] = pt;
                }
                objPointToTrack[objIdx] = merged;

                if (objPointToJoint[objIdx]) {
                    delete objPointToJoint[objIdx];
                }

                updateFrame();
                updateSelection();
                if (onDone) onDone();
            },
            error: function(xhr) {
                if (onError) onError(xhr);
            }
        });
    }

    // Helper: clear 2D track for an object point only from
    // a given frame onward, keeping earlier frames intact.
    function clearTrackFromFrame(objIdx, fromFrame) {
        const track = objPointToTrack[objIdx];
        if (!track) return;

        const pruned = {};
        for (const [fStr, pt] of Object.entries(track)) {
            const fi = parseInt(fStr);
            if (Number.isNaN(fi) || fi < fromFrame) {
                pruned[fStr] = pt;
            }
        }

        if (Object.keys(pruned).length > 0) {
            objPointToTrack[objIdx] = pruned;
        } else {
            delete objPointToTrack[objIdx];
        }
    }

    // 2D Tracking Logic - single point
    $('#btn-track-2d').click(function() {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first.");
            return;
        }

        const info = pending2DPoints[activeObjectPointIndex];
        if (!info) {
            alert("Please click on the video frame to set a 2D point for this object point first.");
            return;
        }

        const btn = $(this);
        const btnAll = $('#btn-track-2d-all');
        btn.prop('disabled', true).text('Tracking...');
        btnAll.prop('disabled', true);
        $('#2d-status').text("Tracking in progress...");

        const objIdx = activeObjectPointIndex;
        runTrackingForObject(
            objIdx,
            info.frame,
            info.x,
            info.y,
            function() {
                delete pending2DPoints[objIdx];
                pending2DPoint = null;
                if (Object.keys(pending2DPoints).length === 0) {
                    btnAll.prop('disabled', true).text('Track All');
                }
                btn.prop('disabled', false).text('Track 2D Point');
                $('#2d-status').text("Tracking complete!");
                update2DOverlay();
            },
            function(xhr) {
                btn.prop('disabled', false).text('Track 2D Point');
                btnAll.prop('disabled', Object.keys(pending2DPoints).length === 0).text('Track All');
                $('#2d-status').text("Error: " + (xhr.responseJSON?.error || "Tracking failed"));
                alert("Tracking failed: " + (xhr.responseJSON?.error || "Unknown error"));
            }
        );
    });

    // 2D Tracking Logic - track all pending points using a single CoTracker call per frame
    $('#btn-track-2d-all').click(function() {
        const entries = Object.entries(pending2DPoints);
        if (entries.length === 0) {
            alert("No pending 2D points to track.");
            return;
        }

        const btnAll = $(this);
        const btnSingle = $('#btn-track-2d');
        btnAll.prop('disabled', true).text('Tracking All...');
        btnSingle.prop('disabled', true);
        $('#2d-status').text("Tracking all pending 2D points...");

        // Group pending points by their start frame so that
        // each CoTracker run can handle multiple queries at once.
        const frameGroups = {};
        for (const [objIdxStr, info] of entries) {
            const f = info.frame;
            if (!frameGroups[f]) {
                frameGroups[f] = [];
            }
            frameGroups[f].push({
                objIdx: parseInt(objIdxStr),
                x: info.x,
                y: info.y
            });
        }

        const frames = Object.keys(frameGroups)
            .map(f => parseInt(f))
            .sort((a, b) => a - b);

        let groupIndex = 0;
        let hadError = false;

        function processNextGroup() {
            if (groupIndex >= frames.length) {
                // All groups processed
                pending2DPoints = {};
                pending2DPoint = null;
                btnAll.prop('disabled', true).text('Track All');
                btnSingle.prop('disabled', false).text('Track 2D Point');
                $('#2d-status').text(hadError ? 'Finished with some errors' : 'All tracking complete!');
                updateFrame();
                updateSelection();
                update2DOverlay();
                return;
            }

            const frame = frames[groupIndex++];
            const points = frameGroups[frame];
            if (!points || points.length === 0) {
                processNextGroup();
                return;
            }

            $.ajax({
                url: 'api/track_2d_multi',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    frame_idx: frame,
                    points: points.map(p => ({ obj_idx: p.objIdx, x: p.x, y: p.y }))
                }),
                success: function(response) {
                    const tracksByObj = response.tracks || {};
                    for (const [objIdxKey, trackDict] of Object.entries(tracksByObj)) {
                        const objIdx = parseInt(objIdxKey);
                        const existing = objPointToTrack[objIdx] || {};
                        const merged = { ...existing };

                        for (const [fStr, pt] of Object.entries(trackDict)) {
                            const fi = parseInt(fStr);
                            if (!Number.isNaN(fi) && fi >= frame) {
                                merged[fi] = pt;
                            }
                        }

                        objPointToTrack[objIdx] = merged;

                        if (objPointToJoint[objIdx]) {
                            delete objPointToJoint[objIdx];
                        }

                        if (pending2DPoints[objIdx] !== undefined) {
                            delete pending2DPoints[objIdx];
                        }
                    }

                    // Update view after each group
                    updateFrame();
                    updateSelection();
                    update2DOverlay();

                    processNextGroup();
                },
                error: function(xhr) {
                    console.error('Multi-point tracking error for frame', frame, xhr.responseJSON || xhr.statusText);
                    hadError = true;
                    processNextGroup();
                }
            });
        }

        // Start processing frame groups sequentially
        processNextGroup();
    });

    // Canvas Click Handler (2D Selection)
    $('#modal-video-overlay').click(function(e) {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first!");
            return;
        }

        const rect = this.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        const img = document.getElementById('modal-video-frame');
        
        // Calculate actual image dimensions and offsets due to object-fit: contain
        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;
        
        let renderWidth, renderHeight, offsetX, offsetY;
        
        if (clientRatio > naturalRatio) {
            // Image is pillarboxed (black bars on sides)
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            // Image is letterboxed (black bars on top/bottom)
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }
        
        // Check if click is within the actual image
        if (clickX < offsetX || clickX > offsetX + renderWidth ||
            clickY < offsetY || clickY > offsetY + renderHeight) {
            // Clicked on black bars
            return;
        }
        
        // Map to natural coordinates
        const x = (clickX - offsetX) * (img.naturalWidth / renderWidth);
        const y = (clickY - offsetY) * (img.naturalHeight / renderHeight);
        
        // Store this click as a pending 2D point for the active object index
        pending2DPoints[activeObjectPointIndex] = {
            x: x,
            y: y,
            displayX: clickX,
            displayY: clickY,
            frame: currentFrame
        };
        // Keep last clicked for legacy use
        pending2DPoint = pending2DPoints[activeObjectPointIndex];

        const pendingCount = Object.keys(pending2DPoints).length;
        $('#2d-status').text(
            `Selected 2D for Obj Point ${activeObjectPointIndex}: (${Math.round(x)}, ${Math.round(y)}). ` +
            `Pending points: ${pendingCount}`
        );
        $('#btn-track-2d').prop('disabled', false);
        $('#btn-track-2d-all').prop('disabled', false);

        // Redraw overlay to show all pending points
        update2DOverlay();
    });
    
    // Human Joint Mapping Logic
    $('#btn-set-joint').click(function() {
        if (activeObjectPointIndex === -1) {
            alert("Please select a 3D object point first!");
            return;
        }
        if (!selectedJointName) {
            alert("Please select a joint from the dropdown first!");
            return;
        }
        
        // Add a joint mapping keyframe for this object from the
        // current frame onward (earlier frames remain unchanged).
        addJointKeyframe(activeObjectPointIndex, currentFrame, selectedJointName);
        applyJointMappingForCurrentFrame();
        
        // Clear any 2D track for this point (exclusive choice)
        if (objPointToTrack[activeObjectPointIndex]) {
            delete objPointToTrack[activeObjectPointIndex];
        }
        
        $('#selected-joint-display').text(selectedJointName + " (Linked)");
        updateSelection(); // Update 3D view
    });

    // Modal Controls
    $('#btn-annotate').click(openModal);
    $('#close-modal').click(closeModal);
    $(window).click(function(event) {
        if (event.target.id === 'annotation-modal') {
            closeModal();
        }
    });

    function openModal() {
        if (!hasCheckedScale) {
            alert('Please open "Check Scale" and review the object before annotating.');
            return;
        }
        // Pause video if playing
        if (isPlaying) {
            togglePlay();
        }
        
        $('#modal-frame-idx').text(currentFrame);
        $('#annotation-modal').show();
        
        // Initialize 2D canvas
        init2DCanvas();

        // Ensure the 2D view image and overlay are synced to current frame
        // Call updateFrame once to set src for modal-video-frame and draw 2D overlay
        updateFrame();
        
        // Trigger Plotly resize/redraw
        if (meshData) {
            Plotly.relayout('3d-viewer', {
                'width': $('#3d-viewer').width(),
                'height': $('#3d-viewer').height()
            });
        }
    }

    function closeModal() {
        $('#annotation-modal').hide();
    }

    function fetchMetadata() {
        $('#scene-status').text('Scene: loading metadata...');
        $.get('api/metadata', function(data) {
            totalFrames = data.total_frames;
            fps = data.fps || 30; // Use original video fps
            $('#frame-slider').attr('max', totalFrames - 1);
            
            // Update Info Panel
            $('#info-video').text(`Video: ${data.video_name}`);
            $('#info-object').text(`Object: ${data.obj_name}`);
            $('#info-res').text(`Res: ${data.width}x${data.height}`);
            $('#info-fps').text(`FPS: ${Math.round(fps)}`);
            $('#info-frames').text(`Frames: ${totalFrames}`);
            
            // Preload the first frame
            $('#scene-status').text('Scene: loading frame 0...');
            loadFrame(0);
            
            // Preload subsequent frames
            preloadFrames(0, Math.min(5, totalFrames - 1));
            $('#scene-status').text('Scene: ready');
        }).fail(function(xhr) {
            console.warn('Failed to fetch metadata:', xhr.responseJSON || xhr.statusText);
            $('#scene-status').text('Scene: failed to load metadata');
        });
    }
    
    function preloadFrames(startFrame, endFrame) {
        // Preload frames in background with caching for better performance
        const framesToPreload = [];

        // Collect frames that aren't already cached
        for (let i = startFrame; i <= endFrame; i++) {
            if (!preloadCache.has(i)) {
                framesToPreload.push(i);
            }
        }

        // Limit cache size to prevent memory leaks (keep last 50 frames)
        if (preloadCache.size > 50) {
            const sortedKeys = Array.from(preloadCache.keys()).sort((a, b) =>
                Math.abs(a - currentFrame) - Math.abs(b - currentFrame)
            );
            // Remove oldest frames beyond the cache limit
            sortedKeys.slice(30).forEach(key => {
                preloadCache.delete(key);
            });
        }

        // Preload new frames
        framesToPreload.forEach(frameNum => {
            const img = new Image();
            img.onload = () => {
                // Cache the loaded image
                preloadCache.set(frameNum, img);
            };
            img.src = 'api/frame/' + frameNum + '?v=' + frameCacheKey;
        });
    }
    
    // New Controls
    $('#static-object').change(function() {
        const isStatic = $(this).is(':checked');
        console.log("Static Object mode:", isStatic);
        // TODO: Send this state to backend if needed for tracking logic
    });

    // ---------------- HOI annotation task related (progress 2.0 list + start/finish buttons) ----------------

    // Keep the most recently fetched HOI task list (progress=2.0)
    let lastHoiTasks = [];

    function loadHoiTasks() {
        return $.getJSON('api/hoi_tasks', function(resp) {
            const list = resp.tasks || [];
            lastHoiTasks = list;
            renderHoiList(list);
        }).fail(function(xhr) {
            console.error('Failed to load HOI tasks:', xhr.responseJSON || xhr.statusText);
            $('#hoi-status').text('Failed to load annotation list');
        });
    }

    function renderHoiList(list) {
        const container = $('#hoi-video-list');
        container.empty();

        if (!list.length) {
            container.append('<div style="padding: 8px 10px; font-size: 13px; color: #666;">No videos with progress=2.0</div>');
            return;
        }

        list.forEach(function(rec, idx) {
            const fileName = rec.file_name || rec.file_path || 'unknown';
            const objectCategory = rec.object_category || '-';
            const sf = rec.session_folder || '';
            const locked = !!rec._locked;
            const lockedBy = rec._locked_by;
            const lockedByMe = !!rec._locked_by_me;

            let lockText = '';
            if (locked) {
                lockText = lockedByMe ? 'My annotation' : ('Annotating: ' + (lockedBy || 'unknown'));
            }

            const item = $(`
                <div class="hoi-item" data-session-folder="${sf}"
                     style="padding: 8px 10px; border-bottom: 1px solid #eee; cursor: pointer; font-size: 13px; background: #fafafa;">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                        <div style="font-weight: 600; color: #333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            ${fileName}
                        </div>
                        ${lockText ? `<div style="font-size: 11px; padding: 2px 6px; border-radius: 999px; white-space: nowrap; ${lockedByMe ? 'background:#e7f5ff;color:#1971c2;border:1px solid #a5d8ff;' : 'background:#fff5f5;color:#c92a2a;border:1px solid #ffc9c9;'}">${lockText}</div>` : ''}
                    </div>
                    <div style="font-size: 12px; color: #777; display: flex; justify-content: space-between; gap: 6px;">
                        <span>Obj: ${objectCategory}</span>
                        <span style="max-width: 55%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${sf}</span>
                    </div>
                </div>
            `);

            item.on('click', function() {
                selectedSessionFolder = sf;
                $('#hoi-video-list .hoi-item').css('background', '#fafafa');
                $(this).css('background', '#e5e9ff');
                $('#hoi-status').text(`Selected: ${fileName}`);
            });

            container.append(item);
        });
    }

    function startHoiSession(sessionFolder) {
        if (!currentUser) {
            promptLogin(function(ok) {
                if (ok) {
                    $('#hoi-status').text('Logged in, please click Start Annotation again');
                }
            });
            return;
        }
        if (!sessionFolder) {
            $('#hoi-status').text('Please select a video from the list first');
            return;
        }

        selectedSessionFolder = sessionFolder;
        $('#hoi-status').text('Loading video and scene data...');
        $('#scene-status').text('Scene: initializing...');
        $.ajax({
            url: 'api/hoi_start',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ session_folder: selectedSessionFolder }),
            success: function(resp) {
                currentSessionFolder = selectedSessionFolder;
                frameCacheKey = Date.now(); // bust cached frames for new session
                preloadCache.clear();

                // IMPORTANT: fully reset client-side state so the next session is clean
                // (avoid leftover 2D points, old tracks/keyframes, stuck modes/modals, etc.)
                resetClientStateForNewSession();

                hasCheckedScale = false; // force scale review for each new session
                lastAppliedScale = 1.0;
                resetScaleViewerTransform();
                $('#hoi-status').text('Loaded successfully, refreshing video and scene...');
                $('#scene-status').text('Scene: loading metadata & first frame...');
                fetchMetadata();
                loadMesh();
                // Refresh list so lock owner display is up-to-date
                loadHoiTasks();
            },
            error: function(xhr) {
                const resp = xhr.responseJSON || {};
                const msg = resp.error || xhr.statusText;

                // If the video has already been annotated by someone else, auto-refresh the list and skip to the next one
                if (resp.already_annotated) {
                    $('#hoi-status').text('This video is already annotated, moving to next...');
                    loadHoiTasks().always(function() {
                        const nextSf = pickNextHoiSession(sessionFolder);
                        if (nextSf) {
                            selectedSessionFolder = nextSf;
                            startHoiSession(nextSf);
                        } else {
                            $('#hoi-status').text('No more videos to annotate');
                            $('#scene-status').text('Scene: idle');
                        }
                    });
                    return;
                }

                $('#hoi-status').text('Failed to start annotation: ' + msg);
                $('#scene-status').text('Scene: error – ' + msg);
                // Refresh list to show lock owner
                loadHoiTasks();
            }
        });
    }

    function clearCanvasById(id) {
        const canvas = document.getElementById(id);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function resetClientStateForNewSession() {
        // Per-session defaults
        // Static Object should NOT carry over to the next case.
        if ($('#static-object').length) {
            $('#static-object').prop('checked', false);
        }

        // Stop playback/timers
        isPlaying = false;
        if (playInterval) {
            clearInterval(playInterval);
            playInterval = null;
        }
        if (playAnimationFrameId !== null) {
            cancelAnimationFrame(playAnimationFrameId);
            playAnimationFrameId = null;
        }
        if (sliderUpdateTimeout) {
            clearTimeout(sliderUpdateTimeout);
            sliderUpdateTimeout = null;
        }
        if (dragDebounceTimer) {
            clearTimeout(dragDebounceTimer);
            dragDebounceTimer = null;
        }
        $('#play-pause').text('▶ Play').removeClass('playing');

        // Close any open modals to avoid overlay residue
        $('#annotation-modal').hide();
        $('#magnify-modal').hide();
        $('#scale-modal').hide();

        // Reset annotation data
        totalFrames = 0;
        currentFrame = 0;
        selectedPoints = [];
        activeObjectPointIndex = -1;
        humanKeypoints = {};
        objPointToJoint = {};
        objPointToTrack = {};
        jointKeyframesByObj = {};
        visibilityKeyframesByObj = {};
        pending2DPoint = null;
        pending2DPoints = {};
        selectedJointName = null;

        // Reset UI bits
        currentMode = 'view';
        $('.mode-btn').removeClass('active');
        $('#mode-view').addClass('active');
        $('#selected-joint-display').text('');
        $('#2d-status').text('');
        $('#btn-track-2d').prop('disabled', true);
        $('#btn-track-2d-all').prop('disabled', true);

        // Clear overlay canvases immediately
        clearCanvasById('main-video-overlay');
        clearCanvasById('modal-video-overlay');

        // Reset scale viewer cached scene so it can't carry over
        scaleViewerState.baseHuman = null;
        scaleViewerState.baseObject = null;
        scaleViewerState.baseScaleFactor = 1.0;
        scaleViewerState.baseDiag = 1;
        scaleViewerState.layout = null;
        scaleViewerState.activeMode = null;
        scaleViewerState.lastMouse = null;
        scaleViewerState.camera = {
            eye: { x: 1.25, y: 1.25, z: 1.25 },
            up: { x: 0, y: 0, z: 1 },
            center: { x: 0, y: 0, z: 0 }
        };
        $('#scale-status').text('');

        // Reset Plotly state so old traces can't be re-used
        meshData = null;
        meshTrace = null;
        scatterTrace = null;
        humanTrace = null;
        layout = null;

        try {
            const gd3d = document.getElementById('3d-viewer');
            if (gd3d) Plotly.purge(gd3d);
        } catch (e) {
            // ignore
        }

        try {
            const gdScale = document.getElementById('scale-viewer');
            if (gdScale) Plotly.purge(gdScale);
        } catch (e) {
            // ignore
        }
    }

    function pickNextHoiSession(prevSessionFolder) {
        // Prefer next item after prev in the last list; fall back to first unlocked.
        const list = lastHoiTasks || [];
        if (!list.length) return null;

        const prevIdx = list.findIndex(r => (r.session_folder || '') === (prevSessionFolder || ''));
        const candidates = [];
        if (prevIdx >= 0) {
            for (let i = prevIdx + 1; i < list.length; i++) candidates.push(list[i]);
            for (let i = 0; i <= prevIdx; i++) candidates.push(list[i]);
        } else {
            for (let i = 0; i < list.length; i++) candidates.push(list[i]);
        }

        // Prefer unlocked tasks
        const nextUnlocked = candidates.find(r => !r._locked);
        if (nextUnlocked) return nextUnlocked.session_folder;

        // If all locked, return null (don't auto-steal)
        return null;
    }

    $('#hoi-search').on('input', function() {
        const q = $(this).val().trim().toLowerCase();
        $('#hoi-video-list .hoi-item').each(function() {
            const text = $(this).text().toLowerCase();
            $(this).toggle(text.indexOf(q) !== -1);
        });
    });

    $('#btn-hoi-start').on('click', function() {
        startHoiSession(selectedSessionFolder);
    });

    // Quick save button
    $('#btn-save-merged').on('click', function() {
        const btn = $(this);
        btn.prop('disabled', true);
        const prevSession = currentSessionFolder || selectedSessionFolder;
        saveMergedAnnotations(function(ok) {
            if (!ok) {
                btn.prop('disabled', false);
                $('#hoi-status').text('Save failed, please retry');
                return;
            }

            $('#hoi-status').text('Annotation saved, preparing to move to next video...');

            // Release lock for current session (best-effort)
            const finishReq = prevSession
                ? $.ajax({
                    url: 'api/hoi_finish',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ session_folder: prevSession })
                })
                : null;

            const afterRelease = () => {
                // Refresh list and auto-start next
                loadHoiTasks().always(function() {
                    const nextSf = pickNextHoiSession(prevSession);
                    btn.prop('disabled', false);

                    if (!nextSf) {
                        $('#hoi-status').text('Saved: No next video available (may be locked or completed)');
                        return;
                    }

                    // Select & start next
                    selectedSessionFolder = nextSf;
                    startHoiSession(nextSf);
                });
            };

            if (finishReq) {
                finishReq.always(afterRelease);
            } else {
                afterRelease();
            }
        }, { update_progress: true });
    });

    // Delete/Skip current case: mark annotation_progress = -1 and auto-next
    $('#btn-delete-case').on('click', function() {
        const btn = $(this);
        const prevSession = currentSessionFolder || selectedSessionFolder;

        if (!prevSession) {
            alert('No video selected');
            return;
        }

        const ok = confirm('Are you sure you want to delete and skip this case?\n\nThis will set annotation_progress to -1 and automatically move to the next video.');
        if (!ok) return;

        btn.prop('disabled', true);
        $('#hoi-status').text('Marking as deleted and preparing to move...');

        $.ajax({
            url: 'api/hoi_mark_deleted',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ session_folder: prevSession })
        }).done(function(resp) {
            if (!resp || resp.ok !== true) {
                $('#hoi-status').text('Delete failed: API returned error');
                btn.prop('disabled', false);
                return;
            }

            // Refresh list and auto-start next
            loadHoiTasks().always(function() {
                const nextSf = pickNextHoiSession(prevSession);
                btn.prop('disabled', false);

                if (!nextSf) {
                    $('#hoi-status').text('Deleted: No next video available (may be locked or completed)');
                    return;
                }

                selectedSessionFolder = nextSf;
                startHoiSession(nextSf);
            });
        }).fail(function(xhr) {
            const msg = (xhr && xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : 'Request failed';
            $('#hoi-status').text('Delete failed: ' + msg);
            btn.prop('disabled', false);
        });
    });

    // Re-annotate: clear cache and reload video
    $('#btn-reannotate').on('click', function() {
        const btn = $(this);
        const session = currentSessionFolder || selectedSessionFolder;

        if (!session) {
            alert('No video selected');
            return;
        }

        const ok = confirm('Are you sure you want to re-annotate?\n\nThis will clear all annotation cache (kp_record directory and kp_record_merged.json) and reload the video.');
        if (!ok) return;

        btn.prop('disabled', true);
        $('#hoi-status').text('Clearing annotation cache and reloading...');

        $.ajax({
            url: 'api/hoi_reannotate',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ session_folder: session })
        }).done(function(resp) {
            if (!resp || resp.status !== 'success') {
                $('#hoi-status').text('Re-annotate failed: API returned error');
                btn.prop('disabled', false);
                return;
            }

            $('#hoi-status').text('Cache cleared, reloading...');

            // Reset UI state
            currentFrame = 0;
            frameAnnotations = {};
            annotationMap = {};

            // Reload video info and first frame
            loadVideoInfo();
            loadFrame(0);

            $('#hoi-status').text('Re-annotate preparation complete');
            btn.prop('disabled', false);
        }).fail(function(xhr) {
            const msg = (xhr && xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : 'Request failed';
            $('#hoi-status').text('Re-annotate failed: ' + msg);
            btn.prop('disabled', false);
        });
    });

    // Scale slider debounce timer
    let scaleSliderDebounceTimer = null;

    function resetScaleViewerTransform() {
        scaleViewerState.transform = { tx: 0, ty: 0, tz: 0, yaw: 0, pitch: 0, roll: 0, viewRoll: 0 };
        scaleViewerState.activeMode = null;
        scaleViewerState.lastMouse = null;
    }

    function normalizeVec(v) {
        const n = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) || 1;
        return [v[0]/n, v[1]/n, v[2]/n];
    }

    function crossVec(a, b) {
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ];
    }

    function applyViewAlignedTranslation(dx, dy, step) {
        const cam = scaleViewerState.camera || {};
        const eye = cam.eye || { x: 1, y: 1, z: 1 };
        const center = cam.center || { x: 0, y: 0, z: 0 };
        const up = cam.up || { x: 0, y: 0, z: 1 };

        const view = normalizeVec([center.x - eye.x, center.y - eye.y, center.z - eye.z]);
        const upv = normalizeVec([up.x, up.y, up.z]);
        const right = normalizeVec(crossVec(upv, view));
        const screenUp = normalizeVec(crossVec(view, right));

        // Screen coords: x right, y down → use -dy for up direction
        const tx = right[0] * (-dx * step) + screenUp[0] * (-dy * step);
        const ty = right[1] * (-dx * step) + screenUp[1] * (-dy * step);
        const tz = right[2] * (-dx * step) + screenUp[2] * (-dy * step);

        scaleViewerState.transform.tx += tx;
        scaleViewerState.transform.ty += ty;
        scaleViewerState.transform.tz += tz;
    }

    function computeObjectDiag(obj) {
        if (!obj || !obj.x || obj.x.length === 0) return 1;
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < obj.x.length; i++) {
            const x = obj.x[i];
            const y = obj.y[i];
            const z = obj.z[i];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const dx = maxX - minX;
        const dy = maxY - minY;
        const dz = maxZ - minZ;
        return Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6);
    }

    function computeBounds3D(obj) {
        if (!obj || !obj.x || obj.x.length === 0) {
            return { minX: -1, maxX: 1, minY: -1, maxY: 1, minZ: -1, maxZ: 1 };
        }
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < obj.x.length; i++) {
            const x = obj.x[i];
            const y = obj.y[i];
            const z = obj.z[i];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        return { minX, maxX, minY, maxY, minZ, maxZ };
    }

    function mergeBounds3D(a, b) {
        return {
            minX: Math.min(a.minX, b.minX),
            maxX: Math.max(a.maxX, b.maxX),
            minY: Math.min(a.minY, b.minY),
            maxY: Math.max(a.maxY, b.maxY),
            minZ: Math.min(a.minZ, b.minZ),
            maxZ: Math.max(a.maxZ, b.maxZ)
        };
    }

    function transformObjectMesh(baseObj) {
        if (!baseObj) return { x: [], y: [], z: [], i: [], j: [], k: [] };
        const { tx, ty, tz, yaw, pitch, roll, viewRoll } = scaleViewerState.transform;

        const cy = Math.cos(yaw); const sy = Math.sin(yaw);
        const cp = Math.cos(pitch); const sp = Math.sin(pitch);
        const cr = Math.cos(roll); const sr = Math.sin(roll);

        // View-axis rotation (Rodrigues) using current camera view vector
        const cam = scaleViewerState.camera || { eye: { x: 1, y: 1, z: 1 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
        const viewAxis = normalizeVec([
            cam.center.x - cam.eye.x,
            cam.center.y - cam.eye.y,
            cam.center.z - cam.eye.z
        ]);
        const cosV = Math.cos(viewRoll);
        const sinV = Math.sin(viewRoll);

        // Rotation order: Rz(roll) * Rx(pitch) * Ry(yaw) — lightweight but sufficient for preview
        const outX = new Array(baseObj.x.length);
        const outY = new Array(baseObj.y.length);
        const outZ = new Array(baseObj.z.length);

        for (let idx = 0; idx < baseObj.x.length; idx++) {
            let x = baseObj.x[idx];
            let y = baseObj.y[idx];
            let z = baseObj.z[idx];

            // yaw (Y axis)
            const xYaw = cy * x + sy * z;
            const zYaw = -sy * x + cy * z;

            // pitch (X axis)
            const yPitch = cp * y - sp * zYaw;
            const zPitch = sp * y + cp * zYaw;

            // roll (Z axis)
            const xRoll = cr * xYaw - sr * yPitch;
            const yRoll = sr * xYaw + cr * yPitch;

            // View-axis rotation applied after base rotations
            const kx = viewAxis[0];
            const ky = viewAxis[1];
            const kz = viewAxis[2];
            const dot = kx * xRoll + ky * yRoll + kz * zPitch;
            const crossX = ky * zPitch - kz * yRoll;
            const crossY = kz * xRoll - kx * zPitch;
            const crossZ = kx * yRoll - ky * xRoll;

            const xView = xRoll * cosV + crossX * sinV + kx * dot * (1 - cosV);
            const yView = yRoll * cosV + crossY * sinV + ky * dot * (1 - cosV);
            const zView = zPitch * cosV + crossZ * sinV + kz * dot * (1 - cosV);

            outX[idx] = xView + tx;
            outY[idx] = yView + ty;
            outZ[idx] = zView + tz;
        }

        return { x: outX, y: outY, z: outZ, i: baseObj.i, j: baseObj.j, k: baseObj.k };
    }

    function updateScaleViewerPlot() {
        if (!scaleViewerState.baseHuman || !scaleViewerState.baseObject) return;

        const humanTrace = {
            type: 'mesh3d',
            x: scaleViewerState.baseHuman.x,
            y: scaleViewerState.baseHuman.y,
            z: scaleViewerState.baseHuman.z,
            i: scaleViewerState.baseHuman.i,
            j: scaleViewerState.baseHuman.j,
            k: scaleViewerState.baseHuman.k,
            color: 'pink', opacity: 1.0, name: 'Human'
        };

        const transformed = transformObjectMesh(scaleViewerState.baseObject);
        const objectTrace = {
            type: 'mesh3d',
            x: transformed.x, y: transformed.y, z: transformed.z,
            i: transformed.i, j: transformed.j, k: transformed.k,
            opacity: 1.0, name: 'Object'
        };

        // If vertex colors are available, use vertexcolor; otherwise use light blue
        if (scaleViewerState.objectVertexColors && scaleViewerState.objectVertexColors.length > 0) {
            objectTrace.vertexcolor = scaleViewerState.objectVertexColors;
        } else {
            objectTrace.color = 'lightblue';
        }

        const layout = scaleViewerState.layout || {
            scene: { aspectmode: 'data', dragmode: 'orbit' },
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true
        };
        layout.scene = layout.scene || {};
        layout.scene.camera = scaleViewerState.camera || layout.scene.camera;
        scaleViewerState.layout = layout;

        Plotly.react('scale-viewer', [humanTrace, objectTrace], layout, { responsive: true });
    }

    function bindScaleViewerControls() {
        // Plotly graph-level binding must be refreshed after Plotly.newPlot/Plotly.purge.
        // Document-level bindings should only be installed once.
        const plotEl = document.getElementById('scale-viewer');
        if (plotEl && plotEl.on) {
            try {
                if (typeof plotEl.removeAllListeners === 'function') {
                    plotEl.removeAllListeners('plotly_relayout');
                }
            } catch (e) {
                // ignore
            }

            plotEl.on('plotly_relayout', function(e) {
                const cam = e?.['scene.camera'];
                if (cam) {
                    scaleViewerState.camera = {
                        eye: cam.eye || scaleViewerState.camera.eye,
                        up: cam.up || scaleViewerState.camera.up,
                        center: cam.center || scaleViewerState.camera.center
                    };
                }
            });
        }

        if (scaleViewerDocControlsBound) return;

        $(document).on('keydown', function(e) {
            if (!$('#scale-modal').is(':visible')) return;
            const key = (e.key || '').toLowerCase();

            if (key === 'escape') {
                scaleViewerState.activeMode = null;
                scaleViewerState.lastMouse = null;
                $('#scale-status').text('');
                return;
            }

            if (['g', 'r', 's'].includes(key)) {
                // Avoid toggling repeatedly due to key auto-repeat
                if (e.repeat) return;

                // Toggle mode on press (more intuitive than hold-to-act)
                if (scaleViewerState.activeMode === key) {
                    scaleViewerState.activeMode = null;
                    scaleViewerState.lastMouse = null;
                    $('#scale-status').text('');
                    return;
                }

                scaleViewerState.activeMode = key;
                scaleViewerState.lastMouse = null;
                const label = key === 's'
                    ? 'Scale (applies immediately, no reload)'
                    : (key === 'g' ? 'Move (preview only, view-aligned plane)' : 'Rotate (preview only)');
                $('#scale-status').text(`${label}: move mouse to adjust; press the same key again (or Esc) to stop.`);
            }
        });

        $(document).on('mousemove', function(e) {
            if (!$('#scale-modal').is(':visible')) return;
            if (!scaleViewerState.activeMode) return;

            if (!scaleViewerState.lastMouse) {
                scaleViewerState.lastMouse = { x: e.clientX, y: e.clientY };
                return;
            }

            const dx = e.clientX - scaleViewerState.lastMouse.x;
            const dy = e.clientY - scaleViewerState.lastMouse.y;
            scaleViewerState.lastMouse = { x: e.clientX, y: e.clientY };

            const diag = scaleViewerState.baseDiag || 1;
            const translateStep = diag * 0.0025;

            if (scaleViewerState.activeMode === 'g') {
                // View-plane translation: map mouse delta to camera-aligned right/up axes
                applyViewAlignedTranslation(dx, dy, translateStep);
                updateScaleViewerPlot();
                return;
            }

            if (scaleViewerState.activeMode === 'r') {
                // Rotate around view axis: map mouse delta to roll about camera view
                const delta = (-dx + -dy) * 0.01; // right/clockwise, down/clockwise for screen coords
                scaleViewerState.transform.viewRoll += delta;
                updateScaleViewerPlot();
                return;
            }

            if (scaleViewerState.activeMode === 's') {
                const currentScale = parseFloat($('#slider-scale-factor').val()) || lastAppliedScale || 1.0;
                const newScale = Math.max(0.1, Math.min(5, currentScale * (1 + dx * 0.003)));
                $('#slider-scale-factor').val(newScale);
                $('#input-scale-factor').val(newScale.toFixed(2));
                $('#slider-scale-value').text(newScale.toFixed(2));

                // Debounce backend apply so mouse move matches manual input effect
                if (scaleSliderDebounceTimer) {
                    clearTimeout(scaleSliderDebounceTimer);
                }
                scaleSliderDebounceTimer = setTimeout(function() {
                    applyScale(newScale, false);
                }, 200);
            }
        });

        scaleViewerDocControlsBound = true;
    }

    // Function to apply scale (updates backend; also used by mouse-scale gesture)
    function applyScale(scale, updateSlider = true) {
        if (!(scale > 0)) {
            $('#scale-status').text('Scale must be a positive number');
            return;
        }

        // Skip if unchanged to reduce calls
        if (Math.abs(scale - lastAppliedScale) < 1e-4) {
            return;
        }

        // Update slider if requested
        if (updateSlider) {
            const clampedScale = Math.max(0.1, Math.min(5, scale));
            $('#slider-scale-factor').val(clampedScale);
            $('#slider-scale-value').text(clampedScale.toFixed(2));
        }

        $('#btn-apply-scale').prop('disabled', true).text('Applying...');
        $('#scale-status').text('Applying scale, please wait...');

        $.ajax({
            url: 'api/set_scale',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ scale_factor: scale }),
            success: function(resp) {
                const applied = parseFloat(resp.scale_factor) || scale;
                lastAppliedScale = applied;
                // Update local base geometry without reloading scene to avoid lag
                if (scaleViewerState.baseObject && scaleViewerState.baseScaleFactor > 0) {
                    const ratio = applied / scaleViewerState.baseScaleFactor;
                    scaleViewerState.baseObject = {
                        x: scaleViewerState.baseObject.x.map(v => v * ratio),
                        y: scaleViewerState.baseObject.y.map(v => v * ratio),
                        z: scaleViewerState.baseObject.z.map(v => v * ratio),
                        i: scaleViewerState.baseObject.i,
                        j: scaleViewerState.baseObject.j,
                        k: scaleViewerState.baseObject.k
                    };
                    scaleViewerState.baseScaleFactor = applied;
                }
                $('#slider-scale-factor').val(applied);
                $('#input-scale-factor').val(applied.toFixed(2));
                $('#slider-scale-value').text(applied.toFixed(2));
                $('#scale-status').text('Scale applied: ' + applied.toFixed(2) + 'x');
                $('#btn-apply-scale').prop('disabled', false).text('Apply Scale');
                updateScaleViewerPlot();
            },
            error: function(xhr) {
                $('#scale-status').text('Error: ' + (xhr.responseJSON?.error || 'Failed to apply scale'));
                $('#btn-apply-scale').prop('disabled', false).text('Apply Scale');
            }
        });
    }

    $('#btn-check-scale').click(function() {
        hasCheckedScale = true;
        $('#scale-frame-idx').text(currentFrame);
        $('#scale-modal').show();
        
        $('#scale-status').text('');
        // Initialize slider and input with last applied value
        const initScale = lastAppliedScale || 1.0;
        $('#input-scale-factor').val(initScale.toFixed(2));
        $('#slider-scale-factor').val(initScale);
        $('#slider-scale-value').text(initScale.toFixed(2));
        loadScaleViewer(currentFrame);
    });
    
    $('#close-scale-modal').click(function() {
        $('#scale-modal').hide();
    });
    
    $(window).click(function(event) {
        if (event.target.id === 'scale-modal') {
            $('#scale-modal').hide();
        }
    });

    function loadScaleViewer(frameIdx) {
        const viewer = document.getElementById('scale-viewer');

        // Show loading
        Plotly.newPlot('scale-viewer', [], {
            title: 'Loading Scene Data...',
            xaxis: { visible: false },
            yaxis: { visible: false }
        });

        $.get('api/scene_data/' + frameIdx, function(data) {
            const human = data.human;
            const object = data.object;
            // Cache base geometry for preview-only transforms
            scaleViewerState.baseHuman = human;
            scaleViewerState.baseObject = object;
            // Save object vertex colors (if available)
            scaleViewerState.objectVertexColors = object.vertex_colors || null;
            scaleViewerState.baseScaleFactor = lastAppliedScale || 1.0;
            scaleViewerState.baseDiag = computeObjectDiag(object);

            // Fix axis ranges to avoid Plotly autorange re-centering on every react.
            // Otherwise preview translations look like they "reset" and are hard to perceive.
            const hb = computeBounds3D(human);
            const ob = computeBounds3D(object);
            const bounds = mergeBounds3D(hb, ob);
            const dx = bounds.maxX - bounds.minX;
            const dy = bounds.maxY - bounds.minY;
            const dz = bounds.maxZ - bounds.minZ;
            const maxSpan = Math.max(dx, dy, dz, 1e-6);
            const pad = Math.max(scaleViewerState.baseDiag || 1, computeObjectDiag(human) || 1) * 0.35;

            // Use a uniform span for all axes so the scene doesn't look squashed
            // when one axis range is much larger than the others.
            const span = maxSpan + pad * 2;
            const cx = (bounds.minX + bounds.maxX) / 2;
            const cy = (bounds.minY + bounds.maxY) / 2;
            const cz = (bounds.minZ + bounds.maxZ) / 2;
            const xr = [cx - span / 2, cx + span / 2];
            const yr = [cy - span / 2, cy + span / 2];
            const zr = [cz - span / 2, cz + span / 2];

            scaleViewerState.layout = {
                // Keep UI state (camera/zoom) stable across Plotly.react calls
                uirevision: 'scale-viewer',
                scene: {
                    // Keep unit aspect; axis ranges below are uniform-span
                    aspectmode: 'cube',
                    dragmode: 'orbit',
                    xaxis: { autorange: false, range: xr },
                    yaxis: { autorange: false, range: yr },
                    zaxis: { autorange: false, range: zr }
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                showlegend: true
            };
            resetScaleViewerTransform();
            updateScaleViewerPlot();
            bindScaleViewerControls();

            // Show the current frame image alongside the 3D preview
            $('#scale-frame-image').attr('src', 'api/frame/' + frameIdx + '?t=' + Date.now());

        }).fail(function(xhr) {
            alert('Error loading scene data: ' + (xhr.responseJSON?.error || 'Unknown error'));
            $('#scale-modal').hide();
        });
    }

    // Slider change handler - sync with input and auto-apply scale
    $('#slider-scale-factor').on('input', function() {
        const scale = parseFloat($(this).val());
        // Update input field
        $('#input-scale-factor').val(scale.toFixed(2));
        // Update display value
        $('#slider-scale-value').text(scale.toFixed(2));
        
        // Debounce auto-apply to avoid too many requests
        if (scaleSliderDebounceTimer) {
            clearTimeout(scaleSliderDebounceTimer);
        }
        scaleSliderDebounceTimer = setTimeout(function() {
            applyScale(scale, false); // Don't update slider (already synced)
        }, 300); // Wait 300ms after user stops dragging
    });

    // Input change handler - sync with slider
    $('#input-scale-factor').on('input change', function() {
        const val = $(this).val();
        if (!val) return;
        const scale = parseFloat(val);
        if (isNaN(scale) || scale <= 0) return;
        
        // Clamp to slider range
        const clampedScale = Math.max(0.1, Math.min(5, scale));
        // Update slider
        $('#slider-scale-factor').val(clampedScale);
        $('#slider-scale-value').text(clampedScale.toFixed(2));
    });

    // Apply button handler
    $('#btn-apply-scale').click(function() {
        const val = $('#input-scale-factor').val();
        if (!val) {
            $('#scale-status').text('Please enter a scale > 0');
            return;
        }
        const scale = parseFloat(val);
        applyScale(scale, true);
    });



    // Magnify View Logic - REMOVED (Integrated into Focus View)
    // function showMagnifyView(cx, cy, cz, traces) { ... }

    $('#close-magnify-modal').click(function() {
        $('#magnify-modal').hide();
    });

    function togglePlay() {
        if (isPlaying) {
            // Stop playback
            isPlaying = false;
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
            if (playAnimationFrameId !== null) {
                cancelAnimationFrame(playAnimationFrameId);
                playAnimationFrameId = null;
            }
            $('#play-pause').text('▶ Play').removeClass('playing');
            // When playback stops, loadFrame will handle UI updates
            // Full 3D view updates happen only when user interacts with 3D view
        } else {
            // Start playback
            // Ensure we're not at the end
            if (currentFrame >= totalFrames - 1) {
                currentFrame = 0;
            }

            isPlaying = true;
            $('#play-pause').text('⏸ Pause').addClass('playing');

            // Update current frame first to ensure display is correct
            loadFrame(currentFrame);

            // Use simple setInterval for playback, similar to test_video
            // This avoids the overhead of requestAnimationFrame for frame-based playback
            const frameDelay = 1000 / fps; // Match test_video: use exact fps timing

            playInterval = setInterval(() => {
                if (!isPlaying) return;

                if (currentFrame < totalFrames - 1) {
                    currentFrame++;
                    loadFrame(currentFrame);
                } else {
                    // Reached end, stop playback
                    togglePlay();
                }
            }, frameDelay);
        }
    }

    function loadMesh() {
        $('#scene-status').text('Scene: loading mesh...');
        $.get('api/mesh', function(data) {
            meshData = data;
            // Save vertex colors (if available)
            meshVertexColors = data.vertex_colors || null;
            renderMesh();
            $('#scene-status').text('Scene: ready');
        }).fail(function(xhr) {
            console.warn('Failed to load mesh:', xhr.responseJSON || xhr.statusText);
            $('#scene-status').text('Scene: mesh not available');
        });
    }

    function renderMesh() {
        meshTrace = {
            type: 'mesh3d',
            x: meshData.x,
            y: meshData.y,
            z: meshData.z,
            i: meshData.i,
            j: meshData.j,
            k: meshData.k,
            opacity: 1.0,
            flatshading: true,
            hoverinfo: 'none',
            name: 'Mesh'
        };

        // If vertex colors are available, use vertexcolor; otherwise use gray
        if (meshVertexColors && meshVertexColors.length > 0) {
            meshTrace.vertexcolor = meshVertexColors;
        } else {
            meshTrace.color = 'lightgray';
        }

        scatterTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: [],
            y: [],
            z: [],
            marker: { size: 8, color: 'red' },
            name: 'Object Points',
            hoverinfo: 'text',
            text: []
        };

        humanTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: [],
            y: [],
            z: [],
            marker: { size: 10, color: 'lime' },
            name: 'Human Joints',
            hoverinfo: 'text',
            text: []
        };

        layout = {
            scene: {
                aspectmode: 'data',
                dragmode: 'orbit'
            },
            margin: { l: 0, r: 0, b: 0, t: 0 },
            showlegend: true,
            legend: { x: 0, y: 1 }
        };

        Plotly.newPlot('3d-viewer', [meshTrace, scatterTrace, humanTrace], layout, {responsive: true});

        document.getElementById('3d-viewer').on('plotly_click', function(data) {
            const point = data.points[0];
            
            if (currentMode === 'view') return;

            // Select Mode: Add new or Select existing
            if (currentMode === 'select') {
                // Helper function to check if ALL selected points have at least one annotation
                // Returns true if all points have 2D tracking OR human keypoint (or both)
                function allPointsHaveAnnotation() {
                    if (selectedPoints.length === 0) return true; // No points selected, allow selection
                    
                    // Check each selected point
                    for (const pt of selectedPoints) {
                        // If a point is hidden from this frame onward (e.g. deleted via Manage Points),
                        // it should not block selecting new points.
                        if (!isObjectVisibleAtFrame(pt.index, currentFrame)) {
                            continue;
                        }

                        const objIdx = pt.index;
                        // Check if this point has 2D point selected (in pending2DPoints) or completed tracking
                        const has2DPointSelected = !!pending2DPoints[objIdx];
                        const has2DTrackCompleted = objPointToTrack[objIdx] && 
                                                    Object.keys(objPointToTrack[objIdx]).length > 0;
                        const has2DTrack = has2DPointSelected || has2DTrackCompleted;
                        // Check if this point has human keypoint mapping
                        const hasHumanKp = !!objPointToJoint[objIdx];
                        
                        // If this point has neither 2D point selected/tracked nor human keypoint, return false
                        if (!has2DTrack && !hasHumanKp) {
                            return false;
                        }
                    }
                    return true; // All points have at least one annotation
                }

                // Click on Mesh (curveNumber 0) -> Add new point
                if (point.curveNumber === 0) {
                    const idx = point.pointNumber;
                    const x = meshData.x[idx];
                    const y = meshData.y[idx];
                    const z = meshData.z[idx];
                    
                    // Check if already selected
                    const existingIdx = selectedPoints.findIndex(p => p.index === idx);
                    if (existingIdx === -1) {
                        // This is a NEW point - check if all previously selected points have annotation
                        if (!allPointsHaveAnnotation()) {
                            alert("Please select 2D tracking or human keypoints first.");
                            return;
                        }
                        // Add new point
                        selectedPoints.push({index: idx, x: x, y: y, z: z});
                        activeObjectPointIndex = idx; // Set as active
                    } else {
                        // Just select existing point - no check needed
                        activeObjectPointIndex = idx;
                    }
                    
                    // Reset pending 2D point when switching active object point
                    pending2DPoint = null;
                    
                    // Defer UI update to prevent freezing the browser event loop
                    setTimeout(updateSelection, 10);
                    
                    // NOTE: intentionally do NOT call updateFrame() here
                    // to avoid triggering 2D redraw on every 3D click.
                }
                
                // Click on Object Point (curveNumber 1) -> Select it
                else if (point.curveNumber === 1) {
                    const idx = point.pointNumber; // Index in selectedPoints array
                    if (idx >= 0 && idx < selectedPoints.length) {
                        activeObjectPointIndex = selectedPoints[idx].index;
                        pending2DPoint = null;
                        setTimeout(updateSelection, 10);
                        // updateFrame();
                    }
                }
            }

            // Delete Mode: Click on Selected Points (curveNumber 1)
            else if (currentMode === 'delete') {
                if (point.curveNumber === 1) { // Object Point
                    const idx = point.pointNumber; // Index in selectedPoints array
                    if (idx >= 0 && idx < selectedPoints.length) {
                        const objIdx = selectedPoints[idx].index;
                        
                        // Check if this point has 2D point selected (in pending2DPoints) or completed tracking
                        const has2DPointSelected = !!pending2DPoints[objIdx];
                        const has2DTrackCompleted = objPointToTrack[objIdx] && 
                                                    Object.keys(objPointToTrack[objIdx]).length > 0;
                        const has2DTrack = has2DPointSelected || has2DTrackCompleted;
                        const hasHumanKp = !!objPointToJoint[objIdx];
                        
                        // If the point has neither 2D point selected/tracked nor human keypoint,
                        // completely remove it from selectedPoints
                        if (!has2DTrack && !hasHumanKp) {
                            // Remove from selectedPoints array
                            selectedPoints.splice(idx, 1);
                            
                            // Clear any pending 2D points for this object
                            if (pending2DPoints[objIdx]) {
                                delete pending2DPoints[objIdx];
                            }
                            
                            // Clear any keyframes for this object
                            if (jointKeyframesByObj[objIdx]) {
                                delete jointKeyframesByObj[objIdx];
                            }
                            if (visibilityKeyframesByObj[objIdx]) {
                                delete visibilityKeyframesByObj[objIdx];
                            }
                            
                            if (activeObjectPointIndex === objIdx) {
                                activeObjectPointIndex = -1;
                                pending2DPoint = null;
                            }
                            
                            setTimeout(updateSelection, 10);
                            return;
                        }
                        
                        // If the point has 2D tracking or human keypoint,
                        // keep the existing logic: hide from current frame onward
                        addJointKeyframe(objIdx, currentFrame, null);
                        addVisibilityKeyframe(objIdx, currentFrame, false);
                        applyJointMappingForCurrentFrame();

                        // For any 2D tracks linked to this object point,
                        // only clear tracking results from the current
                        // frame onward, preserving earlier frames.
                        clearTrackFromFrame(objIdx, currentFrame);

                        if (activeObjectPointIndex === objIdx) {
                            activeObjectPointIndex = -1;
                            pending2DPoint = null;
                        }

                        setTimeout(updateSelection, 10);
                        // Likewise, avoid updateFrame() on delete to isolate 3D performance
                    }
                }
            }
        });
        
        document.getElementById('3d-viewer').oncontextmenu = function() { return false; };
    }

    function updateSelection() {
        // Prepare data for Object Points (Trace 1), respecting
        // per-frame visibility so that points deleted at a
        // given frame disappear from that frame onward.
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        const text = [];

        selectedPoints.forEach(p => {
            if (!isObjectVisibleAtFrame(p.index, currentFrame)) {
                return; // hidden from this frame onward
            }

            x.push(p.x);
            y.push(p.y);
            z.push(p.z);

            // Highlight the selected point in blue; use vertex color (if available) or red for other points
            if (p.index === activeObjectPointIndex) {
                colors.push('blue');
            } else if (meshVertexColors && meshVertexColors[p.index]) {
                colors.push(meshVertexColors[p.index]);
            } else {
                colors.push('red');
            }

            let status = '';
            if (objPointToJoint[p.index]) status = ` (Linked to ${objPointToJoint[p.index]})`;
            else if (objPointToTrack[p.index]) status = ` (Tracked)`;
            text.push(`ID: ${p.index}${status}`);
        });

        // Prepare data for Human Keypoints (Trace 2)
        const linkedObjIndices = Object.keys(objPointToJoint)
            .map(Number)
            .filter(idx => isObjectVisibleAtFrame(idx, currentFrame));
        
        const hx = [];
        const hy = [];
        const hz = [];
        const ht = [];
        
        linkedObjIndices.forEach(idx => {
            const pt = selectedPoints.find(p => p.index === idx);
            if (pt) {
                hx.push(pt.x);
                hy.push(pt.y);
                hz.push(pt.z);
                ht.push(`Linked to: ${objPointToJoint[idx]}`);
            }
        });

        // Update traces directly on the graph div to avoid recreating the plot
        const gd = document.getElementById('3d-viewer');
        if (!gd || !meshTrace) return;

        // Update local trace objects
        scatterTrace.x = x;
        scatterTrace.y = y;
        scatterTrace.z = z;
        scatterTrace.text = text;
        scatterTrace.marker.color = colors;

        humanTrace.x = hx;
        humanTrace.y = hy;
        humanTrace.z = hz;
        humanTrace.text = ht;

        // Use Plotly.react for efficient update that handles data changes correctly
        // Pass the same meshTrace reference to avoid re-processing the mesh
        Plotly.react(gd, [meshTrace, scatterTrace, humanTrace], gd.layout);
    }
    
    function updateProgressDisplay(frameNum = currentFrame) {
        // Only update UI elements for responsive display during dragging
        $('#frame-display').text('Frame: ' + frameNum);
        if (!isDragging) {
            // Only update slider if not dragging to avoid conflict
            $('#frame-slider').val(frameNum);
        }
    }

        // Load a specific frame (used for playback and slider dragging)
        function loadFrame(frameNum) {
            currentFrame = Math.max(0, Math.min(frameNum, totalFrames - 1));

            const videoFrame = $('#video-frame')[0];
            const modalVideoFrame = $('#modal-video-frame')[0];
            const frameSrc = 'api/frame/' + currentFrame + '?v=' + frameCacheKey;

            // Main video area: switch images directly without any dimming/transition effects to avoid flickering
            if (videoFrame) {
                videoFrame.src = frameSrc;
            }

            // If the annotation modal is open, sync update the 2D view image and the right-side reference frame
            if ($('#annotation-modal').is(':visible')) {
                if (modalVideoFrame) {
                    modalVideoFrame.src = frameSrc;
                    modalVideoFrame.onload = function() {
                        update2DOverlay();
                        this.onload = null;
                    };
                }
                const modalRefFrame = document.getElementById('modal-reference-frame');
                if (modalRefFrame) modalRefFrame.src = frameSrc;
            }

            $('#frame-display').text('Frame: ' + currentFrame);
            if ($('#annotation-modal').is(':visible')) {
                $('#modal-frame-idx').text(currentFrame);
            }
            if (!isDragging) {
                $('#frame-slider').val(currentFrame);
            }

            // Preload adjacent frames
            preloadFrames(currentFrame);

            // Update 3D/2D state on every frame switch:
            // 1) Reconstruct human joint mapping for the current frame based on keyframes
            // 2) Redraw the left-side 3D object points (with per-frame visibility)
            // 3) Update main video and 2D overlay so tracking points follow frame changes
            applyJointMappingForCurrentFrame();
            updateSelection();
            updateMainOverlay();
            update2DOverlay();
        }

        // Frame load error handling
        $('#video-frame').on('error', function() {
            console.error('Frame load failed:', currentFrame);
        });

        // No longer modify opacity on load to avoid flickering
        $('#video-frame').on('load', function() {
            // no-op
        });

    function updateFrame() {
        // Update frame display immediately for responsiveness
        $('#frame-display').text('Frame: ' + currentFrame);
        if (!isDragging) {
            // Only update slider if not dragging to avoid conflict
            $('#frame-slider').val(currentFrame);
        }
        
        // Use stable URL, let the browser and backend handle caching
        const frameSrc = 'api/frame/' + currentFrame + '?v=' + frameCacheKey;
        
        // Update video frame images when user seeks/pauses to view
        const videoFrame = $('#video-frame')[0];
        const modalVideoFrame = $('#modal-video-frame')[0];
        
        // Update main video frame (no longer apply dimming effect)
        if (videoFrame) {
            const currentBaseSrc = videoFrame.src.split('?')[0];
            const newBaseSrc = frameSrc;
            
            // Update src if frame number changed or force reload
            if (currentBaseSrc !== newBaseSrc || !videoFrame.complete) {
                // Update src
                videoFrame.src = frameSrc;
                
                // Handle load completion
                videoFrame.onload = function() {
                    this.onload = null; // Clean up
                };
                
                // Handle load error
                videoFrame.onerror = function() {
                    this.onerror = null; // Clean up
                };
            } else {
                // Force reload even if src looks the same
                videoFrame.src = frameSrc;
            }
        }
        
        // Update modal video frame and reference frame (only if modal is open)
        if ($('#annotation-modal').is(':visible')) {
            if (modalVideoFrame) modalVideoFrame.src = frameSrc;
            const modalRefFrame = document.getElementById('modal-reference-frame');
            if (modalRefFrame) modalRefFrame.src = frameSrc;
        }

        // Update overlays on main video and 2D view (tracking points, pending points)
        updateMainOverlay();
        update2DOverlay();
    }

    // Draw tracking points on the main video
    function updateMainOverlay() {
        const canvas = document.getElementById('main-video-overlay');
        const img = document.getElementById('video-frame');
        if (!canvas || !img) return;

        // Ensure the image is fully loaded before drawing
        if (!img.complete || img.naturalWidth === 0) {
            img.onload = function() {
                updateMainOverlay();
                img.onload = null;
            };
            return;
        }

        // Resize canvas to match the current display size
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;

        let renderWidth, renderHeight, offsetX, offsetY;
        if (clientRatio > naturalRatio) {
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }

        // Draw all tracking points on the main video (consistent with modal, highlight the currently active point)
        for (const [objIdx, tracks] of Object.entries(objPointToTrack)) {
            const pt = tracks[currentFrame] || tracks[String(currentFrame)];
            if (!pt) continue;

            const [x, y] = pt;
            const displayX = x * (renderWidth / img.naturalWidth) + offsetX;
            const displayY = y * (renderHeight / img.naturalHeight) + offsetY;

            ctx.beginPath();
            ctx.arc(displayX, displayY, 4, 0, 2 * Math.PI);

            if (parseInt(objIdx) === activeObjectPointIndex) {
                ctx.fillStyle = '#00ff00';
                ctx.lineWidth = 2;
            } else {
                ctx.fillStyle = '#008800';
                ctx.lineWidth = 1;
            }

            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.stroke();
        }
    }

    // Only responsible for drawing tracking points on the modal canvas based on the current frame
    function update2DOverlay() {
        const canvas = document.getElementById('modal-video-overlay');
        if (!canvas || !$('#annotation-modal').is(':visible')) {
            // Modal not open, skip canvas update
            return;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Pre-calculate image geometry to avoid layout thrashing in loop
        const img = document.getElementById('modal-video-frame');
        if (!img || img.naturalWidth === 0) return; 
        
        const naturalRatio = img.naturalWidth / img.naturalHeight;
        const clientRatio = img.clientWidth / img.clientHeight;
        
        let renderWidth, renderHeight, offsetX, offsetY;
        
        if (clientRatio > naturalRatio) {
            renderHeight = img.clientHeight;
            renderWidth = renderHeight * naturalRatio;
            offsetX = (img.clientWidth - renderWidth) / 2;
            offsetY = 0;
        } else {
            renderWidth = img.clientWidth;
            renderHeight = renderWidth / naturalRatio;
            offsetX = 0;
            offsetY = (img.clientHeight - renderHeight) / 2;
        }

        // Draw tracked points for ALL objects
        for (const [objIdx, tracks] of Object.entries(objPointToTrack)) {
            const pt = tracks[currentFrame] || tracks[String(currentFrame)];
            if (pt) {
                 const [x, y] = pt;
                 
                 const displayX = x * (renderWidth / img.naturalWidth) + offsetX;
                 const displayY = y * (renderHeight / img.naturalHeight) + offsetY;
                 
                 ctx.beginPath();
                 ctx.arc(displayX, displayY, 5, 0, 2 * Math.PI);
                 
                 // Highlight if active
                 if (parseInt(objIdx) === activeObjectPointIndex) {
                     ctx.fillStyle = '#00ff00'; // Lime green for active
                     ctx.lineWidth = 3;
                 } else {
                     ctx.fillStyle = '#008800'; // Darker green for others
                     ctx.lineWidth = 1;
                 }
                 
                 ctx.fill();
                 ctx.strokeStyle = 'white';
                 ctx.stroke();
                 
                 // Label
                 ctx.fillStyle = 'white';
                 ctx.font = '12px Arial';
                 ctx.fillText(`ID: ${objIdx}`, displayX + 8, displayY + 4);
            }
        }
        
        // Draw all pending 2D points (red), one per object index
        for (const [objIdx, p] of Object.entries(pending2DPoints)) {
             let dx = p.displayX;
             let dy = p.displayY;

             if (dx === undefined) {
                 dx = p.x * (renderWidth / img.naturalWidth) + offsetX;
                 dy = p.y * (renderHeight / img.naturalHeight) + offsetY;
             }

             ctx.beginPath();
             ctx.arc(dx, dy, 5, 0, 2 * Math.PI);
             ctx.fillStyle = 'red';
             ctx.fill();
             ctx.strokeStyle = 'white';
             ctx.lineWidth = 2;
             ctx.stroke();
             ctx.fillStyle = 'white';
             ctx.font = '12px Arial';
             ctx.fillText(`Pending ${objIdx}`, dx + 8, dy + 4);
        }
    }

    function saveMergedAnnotations(callback, options) {
        const opts = options || {};

        // Avoid ReferenceError when called before globals are initialized.
        const safeJointKeyframes = (typeof jointKeyframesByObj !== 'undefined' && jointKeyframesByObj) ? jointKeyframesByObj : {};
        const safeVisibilityKeyframes = (typeof visibilityKeyframesByObj !== 'undefined' && visibilityKeyframesByObj) ? visibilityKeyframesByObj : {};
        const safeTracks = (typeof objPointToTrack !== 'undefined' && objPointToTrack) ? objPointToTrack : {};
        const safeTotalFrames = (typeof totalFrames !== 'undefined' && Number.isFinite(totalFrames)) ? totalFrames : null;
        const safeCurrentFrame = (typeof currentFrame !== 'undefined' && Number.isFinite(currentFrame)) ? currentFrame : null;

        // Persist global flags (e.g., Static Object) into merged file.
        // NOTE: The main "Save Annotation" button uses this function, not saveAllAnnotations().
        const hasStaticCheckbox = (typeof $ === 'function') && ($('#static-object').length > 0);
        const isStaticObject = hasStaticCheckbox ? $('#static-object').is(':checked') : false;

        // Prepare payload with all in-memory annotations
        const safeLastFrame = (safeTotalFrames !== null && safeTotalFrames > 0)
            ? (safeTotalFrames - 1)
            : safeCurrentFrame;

        const payload = {
            is_static_object: !!isStaticObject,
            joint_keyframes: safeJointKeyframes,
            visibility_keyframes: safeVisibilityKeyframes,
            tracks: safeTracks,
            total_frames: safeTotalFrames,
            last_frame: safeLastFrame,
            update_progress: !!opts.update_progress
        };

        const req = $.ajax({
            url: 'api/save_merged_annotations',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(payload)
        });

        // Support existing callback usage
        if (callback) {
            req
                .done(function(response) {
                    if (response.status === 'success') {
                        console.log('Merged annotations saved to:', response.path);
                        callback(true);
                    } else {
                        console.error('Failed to save merged annotations:', response);
                        callback(false);
                    }
                })
                .fail(function(xhr, status, error) {
                    console.error('Error saving merged annotations:', error || status);
                    callback(false);
                });
        }

        return req;
    }

    function saveAnnotation() {
        // Construct human_keypoints: { jointName: {index, x, y, z} }
        const humanKeypointsExport = {};
        for (const [objIdx, jointName] of Object.entries(objPointToJoint)) {
            const pt = selectedPoints.find(p => p.index === parseInt(objIdx));
            if (pt) {
                humanKeypointsExport[jointName] = {
                    index: pt.index,
                    x: pt.x, y: pt.y, z: pt.z
                };
            }
        }

        $.ajax({
            url: 'api/save_annotation',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                frame: currentFrame,
                object_points: selectedPoints,
                human_keypoints: humanKeypointsExport,
                tracks: objPointToTrack
            }),
            success: function(response) {
                alert('Saved annotation for frame ' + currentFrame);
            },
            error: function(xhr) {
                alert('Save failed: ' + (xhr.responseJSON?.error || 'Unknown error'));
            }
        });
    }

    function saveAllAnnotations() {
        const isStatic = $('#static-object').is(':checked');

        $.ajax({
            url: 'api/save_merged_annotations',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                is_static_object: isStatic,
                total_frames: totalFrames,
                // Save through the end of sequence by default.
                last_frame: (Number.isFinite(totalFrames) && totalFrames > 0) ? (totalFrames - 1) : currentFrame,
                // Per-object 3D joint keyframes over time
                joint_keyframes: jointKeyframesByObj,
                // Per-object visibility keyframes over time
                visibility_keyframes: visibilityKeyframesByObj,
                // 2D tracks for each object point
                tracks: objPointToTrack
            }),
            success: function(response) {
                const outPath = response.path || 'kp_record_merged.json';
                alert('Saved merged annotations to:\n' + outPath);
            },
            error: function(xhr) {
                alert('Save-all failed: ' + (xhr.responseJSON?.error || 'Unknown error'));
            }
        });
    }
    
    // Management UI
    $('#btn-manage').click(function() {
        renderAnnotationList();
        $('#management-panel').show();
    });
    
    $('#close-manager').click(function() {
        $('#management-panel').hide();
    });

    // ===== AI Auto-Prediction =====
    $('#btn-auto-predict').click(async function() {
        const btn = $(this);
        if (btn.data('running')) {
            return; // Prevent double-clicks
        }

        const confirmed = confirm(
            `Apply AI predictions from frame ${currentFrame} to all subsequent frames?\n\n` +
            `This will ADD predicted contact points to existing annotations.\n` +
            `Existing annotations will be preserved.`
        );
        if (!confirmed) return;

        const originalText = btn.text();
        const restoreButton = () => {
            btn.data('running', false);
            btn.prop('disabled', false);
            btn.text(originalText);
        };

        btn.data('running', true);
        btn.prop('disabled', true);
        btn.text('Predicting...');

        try {
            const resp = await $.ajax({
                url: 'api/auto_predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    frame: currentFrame,
                    threshold: 0.5,
                    top_k: 20  // Limit to top 20 predictions
                })
            });

            if (resp.ok && resp.predictions && resp.predictions.length > 0) {
                // Apply predictions
                const applied = applyIVDPredictions(resp.predictions, currentFrame);

                // Update visualization
                updateSelection();

                alert(`Applied ${applied} AI predictions from frame ${currentFrame}.\n\nPredictions will apply from this frame forward.`);
            } else if (resp.ok && (!resp.predictions || resp.predictions.length === 0)) {
                alert('No contact predictions above threshold. Try lowering the threshold or checking the input data.');
            } else {
                alert('Prediction failed: ' + (resp.error || 'Unknown error'));
            }
        } catch (e) {
            console.error('Auto-predict error:', e);
            let msg = 'Unknown error';
            if (e.responseJSON && e.responseJSON.error) {
                msg = e.responseJSON.error;
            } else if (e.statusText) {
                msg = e.statusText;
            } else if (e.status) {
                msg = `HTTP ${e.status}`;
            } else if (typeof e === 'string') {
                msg = e;
            } else if (e.message) {
                msg = e.message;
            }
            console.error('Error details:', JSON.stringify(e, null, 2));
            alert('AI Prediction failed: ' + msg + '\n\nCheck browser console for details.');
        } finally {
            restoreButton();
        }
    });

    /**
     * Apply IVD model predictions to annotations.
     * @param {Array} predictions - List of predictions with joint, xyz, vertex_idx, confidence
     * @param {number} fromFrame - Frame from which predictions apply
     * @returns {number} Number of predictions applied
     */
    function applyIVDPredictions(predictions, fromFrame) {
        let applied = 0;

        // Safety check for meshData
        if (!meshData || !meshData.x || !meshData.y || !meshData.z) {
            console.error('applyIVDPredictions: meshData not loaded');
            alert('Error: Mesh data not loaded. Please load a video first.');
            return 0;
        }

        for (const pred of predictions) {
            const vertexIdx = pred.vertex_idx;
            const jointName = pred.joint;
            const xyz = pred.xyz;
            const confidence = pred.confidence;

            // Skip if vertex_idx is invalid
            if (vertexIdx === undefined || vertexIdx === null || vertexIdx < 0) {
                console.warn('Invalid vertex_idx in prediction:', pred);
                continue;
            }

            // Skip if vertex_idx is out of bounds
            if (vertexIdx >= meshData.x.length) {
                console.warn('vertex_idx out of bounds:', vertexIdx, 'max:', meshData.x.length - 1);
                continue;
            }

            // Check if this vertex is already a selected point
            let existingPoint = selectedPoints.find(pt => pt.index === vertexIdx);

            if (!existingPoint) {
                // Add new point to selectedPoints
                // meshData has x, y, z as separate arrays
                const vx = meshData.x[vertexIdx];
                const vy = meshData.y[vertexIdx];
                const vz = meshData.z[vertexIdx];

                existingPoint = {
                    index: vertexIdx,
                    x: vx,
                    y: vy,
                    z: vz,
                    confidence: confidence
                };
                selectedPoints.push(existingPoint);

                // Initialize visibility as visible from frame 0
                addVisibilityKeyframe(vertexIdx, 0, true);
            }

            // Add joint keyframe for this prediction from the current frame
            addJointKeyframe(vertexIdx, fromFrame, jointName);

            applied++;
            console.log(`Applied prediction: ${jointName} -> vertex ${vertexIdx} (confidence: ${confidence.toFixed(3)})`);
        }

        // Recompute objPointToJoint for current frame
        applyJointMappingForCurrentFrame();

        return applied;
    }

    function renderAnnotationList() {
        const tbody = $('#annotation-table tbody');
        tbody.empty();
        
        selectedPoints.forEach(pt => {
            const idx = pt.index;
            let type = 'None';
            let target = '-';
            
            if (objPointToJoint[idx]) {
                type = 'Human Joint';
                target = objPointToJoint[idx];
            } else if (objPointToTrack[idx]) {
                type = '2D Track';
                const track = objPointToTrack[idx];
                const currentPos = track[currentFrame] || track[String(currentFrame)];
                target = currentPos ? `Frame ${currentFrame}: (${Math.round(currentPos[0])}, ${Math.round(currentPos[1])})` : 'No track this frame';
            }
            
            const tr = $('<tr>');
            tr.append($('<td>').text(idx));
            tr.append($('<td>').text(type));
            tr.append($('<td>').text(target));

            const actions = $('<td>');

            if (type === 'Human Joint') {
                const editJointBtn = $('<button>').text('Edit Joint').click(function() {
                    activeObjectPointIndex = idx;
                    if (!$('#annotation-modal').is(':visible')) {
                        $('#annotation-modal').show();
                    }
                    $('#tab-human').click();
                    $('#selected-joint-display').text(objPointToJoint[idx] || 'None');
                    updateSelection();
                });
                actions.append(editJointBtn);
            } else if (type === '2D Track') {
                const edit2DBtn = $('<button>').text('Edit 2D').click(function() {
                    activeObjectPointIndex = idx;
                    if (!$('#annotation-modal').is(':visible')) {
                        $('#annotation-modal').show();
                    }
                    $('#tab-2d').click();

                    const track = objPointToTrack[idx] || {};
                    const currentPos = track[currentFrame] || track[String(currentFrame)];
                    if (currentPos) {
                        pending2DPoint = {
                            x: currentPos[0],
                            y: currentPos[1]
                        };
                        $('#2d-status').text(
                            `Current: (${Math.round(currentPos[0])}, ${Math.round(currentPos[1])}). ` +
                            'Click new position on the image, then press "Track 2D Point" to retrack from this frame.'
                        );
                    } else {
                        pending2DPoint = null;
                        $('#2d-status').text(
                            'No 2D point on this frame. Click on the image to choose one, then press "Track 2D Point".'
                        );
                    }

                    $('#btn-track-2d').prop('disabled', false);
                    update2DOverlay();
                });
                actions.append(edit2DBtn);
            }

            const deleteBtn = $('<button>').text('Delete').click(function() {
                // From the management panel, deletion of a Human Joint
                // should only affect from the current frame onward, not
                // earlier frames. We therefore add a null keyframe for
                // this object and hide the 3D point from this frame
                // onward, instead of removing the underlying 3D point
                // globally.
                if (type === 'Human Joint') {
                    addJointKeyframe(idx, currentFrame, null);
                    addVisibilityKeyframe(idx, currentFrame, false);
                    applyJointMappingForCurrentFrame();
                }

                // For any 2D tracks, only clear tracking results from
                // the current frame onward, keeping earlier frames intact.
                clearTrackFromFrame(idx, currentFrame);

                if (activeObjectPointIndex === idx) activeObjectPointIndex = -1;
                updateSelection();
                updateFrame();
                renderAnnotationList();
            });
            actions.append(deleteBtn);

            tr.append(actions);

            tbody.append(tr);
        });
    }

    function togglePoint(idx, x, y, z) {
        // Deprecated in favor of explicit add/delete modes
    }

    let focusMode = 'view'; // 'view', 'magnify'
    let focusTraces = []; // Store traces for magnification

    // Focus Hand Button Handler
    $('#btn-focus-hand').click(function() {
        $('#focus-frame-idx').text(currentFrame);
        $('#focus-modal').show();
        // Reset mode
        focusMode = 'view';
        $('#btn-focus-magnify').css('background-color', '#17a2b8').text('Magnify');
        
        updateFocusView(currentFrame);
    });
    
    // Focus Magnify Button Handler
    $('#btn-focus-magnify').click(function() {
        if (focusMode === 'view') {
            focusMode = 'magnify';
            $(this).css('background-color', '#ffc107').text('Click to Magnify');
        } else {
            focusMode = 'view';
            $(this).css('background-color', '#17a2b8').text('Magnify');
        }
    });
    
    // Focus Reset Button Handler
    $('#btn-focus-reset').click(function() {
        const gd = document.getElementById('focus-viewer');
        // Reset camera to default
        Plotly.relayout(gd, {
            'scene.camera.center': {x: 0, y: 0, z: 0},
            'scene.camera.eye': {x: 1.25, y: 1.25, z: 1.25},
            'scene.camera.up': {x: 0, y: 0, z: 1}
        });
    });

    $('#close-focus-modal').click(function() {
        $('#focus-modal').hide();
    });

    function updateFocusView(frameIdx) {
        const gd = document.getElementById('focus-viewer');
        // Plotly.purge(gd); // Optional: Clear previous

        $.get('api/focus_hand/' + frameIdx, function(data) {
            const human = data.human;
            const object = data.object;
            const camera = data.camera;

            const humanTrace = {
                type: 'mesh3d',
                x: human.x, y: human.y, z: human.z,
                i: human.i, j: human.j, k: human.k,
                color: 'pink', opacity: 1.0,
                name: 'Human'
            };

            const objectTrace = {
                type: 'mesh3d',
                x: object.x, y: object.y, z: object.z,
                i: object.i, j: object.j, k: object.k,
                color: 'lightblue', opacity: 0.8,
                name: 'Object'
            };
            
            focusTraces = [humanTrace, objectTrace];

            const layout = {
                scene: {
                    aspectmode: 'data',
                    camera: camera
                },
                margin: { l: 0, r: 0, b: 0, t: 0 },
                showlegend: true
            };

            Plotly.newPlot('focus-viewer', focusTraces, layout, { responsive: true });
            
            // Attach click handler for magnification
            // Remove previous handlers to avoid duplicates
            gd.removeAllListeners('plotly_click');
            
            gd.on('plotly_click', function(data) {
                if (focusMode === 'magnify') {
                    try {
                        const point = data.points[0];
                        const x = point.x;
                        const y = point.y;
                        const z = point.z;
                        
                        let minX = Infinity, maxX = -Infinity;
                        let minY = Infinity, maxY = -Infinity;
                        let minZ = Infinity, maxZ = -Infinity;
                        
                        [human, object].forEach(mesh => {
                            if (mesh.x && mesh.x.length > 0) {
                                for(let i=0; i<mesh.x.length; i++) {
                                    const v = mesh.x[i]; if(v < minX) minX = v; if(v > maxX) maxX = v;
                                }
                                for(let i=0; i<mesh.y.length; i++) {
                                    const v = mesh.y[i]; if(v < minY) minY = v; if(v > maxY) maxY = v;
                                }
                                for(let i=0; i<mesh.z.length; i++) {
                                    const v = mesh.z[i]; if(v < minZ) minZ = v; if(v > maxZ) maxZ = v;
                                }
                            }
                        });
                        
                        if (!isFinite(minX) || !isFinite(maxX)) {
                            console.error("Invalid mesh bounds");
                            return;
                        }
                        
                        const centerX = (minX + maxX) / 2;
                        const centerY = (minY + maxY) / 2;
                        const centerZ = (minZ + maxZ) / 2;
                        
                        const sizeX = maxX - minX;
                        const sizeY = maxY - minY;
                        const sizeZ = maxZ - minZ;
                        const maxDim = Math.max(sizeX, sizeY, sizeZ) || 1.0; // Avoid div by zero
                        
                        const normX = (x - centerX) / maxDim;
                        const normY = (y - centerY) / maxDim;
                        const normZ = (z - centerZ) / maxDim;
                        
                        // Use _fullLayout to get the actual current camera state
                        const scene = gd._fullLayout ? gd._fullLayout.scene : null;
                        const currentEye = scene ? scene.camera.eye : {x: 1.25, y: 1.25, z: 1.25};
                        
                        const ex = currentEye.x;
                        const ey = currentEye.y;
                        const ez = currentEye.z;
                        const len = Math.sqrt(ex*ex + ey*ey + ez*ez) || 1.0;
                        
                        // Zoom distance: 0.4 units away from the target point
                        const zoomDist = 0.4; 
                        
                        const newEyeX = normX + (ex / len) * zoomDist;
                        const newEyeY = normY + (ey / len) * zoomDist;
                        const newEyeZ = normZ + (ez / len) * zoomDist;
                        
                        Plotly.relayout(gd, {
                            'scene.camera.center': {x: normX, y: normY, z: normZ},
                            'scene.camera.eye': {x: newEyeX, y: newEyeY, z: newEyeZ}
                        });
                        
                        // Reset button state
                        focusMode = 'view';
                        $('#btn-focus-magnify').css('background-color', '#17a2b8').text('Magnify');
                    } catch (e) {
                        console.error("Error in magnify click handler:", e);
                    }
                }
            });

        }).fail(function(xhr) {
            alert('Error loading focus view: ' + (xhr.responseJSON?.error || 'Unknown error'));
            $('#focus-modal').hide();
        });
    }
});
