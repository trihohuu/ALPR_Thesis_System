import numpy as np

def calculate_iou(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    xA = max(x1_a, x1_b)
    yA = max(y1_a, y1_b)
    xB = min(x2_a, x2_b)
    yB = min(y2_a, y2_b)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (x2_a - x1_a) * (y2_a - y1_a)
    boxBArea = (x2_b - x1_b) * (y2_b - y1_b)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class Tracker:
    def __init__(self, iou_threshold=0.5, max_lost=10):
        self.tracks = [] 
        self.track_id_count = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost 
        self.all_tracks = {} 

    def update(self, detections):
        updated_tracks = []
        
        for det in detections:
            # Get info
            box = det['box']
            text = det.get('text', '')
            conf = det.get('ocr_conf', 0.0)
            img = det.get('plate_img', None)
            
            best_match = None
            max_iou = 0
            
            # Compare past tracks with current box
            for track in self.tracks:
                iou = calculate_iou(box, track['box'])
                if iou > max_iou:
                    max_iou = iou
                    best_match = track
            
            if max_iou > self.iou_threshold and best_match:
                det['id'] = best_match['id']
                det['lost_count'] = 0 

                current_best_conf = best_match.get('best_conf', 0.0)
                # Replace if higher conf
                if text != "" and len(text) >= 6 and conf > current_best_conf:
                    best_match['best_conf'] = conf
                    best_match['best_text'] = text
                    best_match['best_img'] = img
                    det['text'] = text
                else:
                    det['text'] = best_match.get('best_text', best_match.get('text', ''))
                
                updated_tracks.append(det)
                best_match['updated'] = True
            else:
                # New plate
                self.track_id_count += 1
                det['id'] = self.track_id_count
                det['lost_count'] = 0
                det['updated'] = True
                
                det['best_conf'] = conf
                det['best_text'] = text
                det['best_img'] = img
                updated_tracks.append(det)

        new_track_list = []
        matched_ids = [t['id'] for t in updated_tracks]
        
        for track in self.tracks:
            if track['id'] not in matched_ids:
                # Increase lost count if not founded in the frame
                track['lost_count'] += 1
                if track['lost_count'] < self.max_lost: # maximum 10 lost-frames
                    new_track_list.append(track)
            else:
                # If found -> Update new coordinates 
                updated_data = next((t for t in updated_tracks if t['id'] == track['id']), None)
                if updated_data:
                    track['box'] = updated_data['box']
                    new_track_list.append(track)

        for track in updated_tracks:
             exists = any(t['id'] == track['id'] for t in new_track_list)
             if not exists:
                 new_track_list.append(track)

        self.tracks = new_track_list
        for track in self.tracks:
            self.all_tracks[track['id']] = track
            
        return updated_tracks