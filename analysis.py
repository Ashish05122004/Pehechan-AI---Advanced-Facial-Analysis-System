import cv2
from deepface import DeepFace
import numpy as np
import time

def preprocess_for_gender_analysis(image):
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()

        if len(processed.shape) == 3:
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
            l_enhanced = clahe.apply(l)

            enhanced_lab = cv2.merge([l_enhanced, a, b])
            processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        height, width = processed.shape[:2]
        if height < 224 or width < 224:
            scale_factor = max(224/height, 224/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return processed
    except Exception as e:
        print(f"Error in gender preprocessing: {e}")
        return image

def analyze_gender_with_multiple_models(image):
    gender_results = []

    preprocessing_variants = [
        preprocess_for_gender_analysis(image),
        preprocess_image_simple(image), 
        image 
    ]

    detector_backends = ['retinaface', 'mtcnn', 'opencv']

    for i, processed_img in enumerate(preprocessing_variants):
        for backend in detector_backends:
            try:
                result = DeepFace.analyze(
                    img_path=processed_img,
                    actions=['gender'],
                    enforce_detection=True,
                    detector_backend=backend,
                    align=True,
                    silent=True
                )

                if isinstance(result, list) and len(result) > 0:
                    result = result[0]

                if result and 'dominant_gender' in result:
                    gender_results.append({
                        'gender': result['dominant_gender'],
                        'confidence': result['gender'][result['dominant_gender']],
                        'all_scores': result['gender'],
                        'preprocessing': i,
                        'backend': backend
                    })

            except Exception as e:
                print(f"Gender analysis failed with preprocessing {i}, backend {backend}: {e}")
                continue

    return gender_results

def get_consensus_gender(gender_results):
    if not gender_results:
        return None

    weighted_scores = {'Man': 0, 'Woman': 0}
    total_weight = 0

    for result in gender_results:
        confidence = result['confidence']
        gender = result['gender']

        weight = confidence / 100.0
        weighted_scores[gender] += weight
        total_weight += weight

    if total_weight == 0:
        return gender_results[0]

    for gender in weighted_scores:
        weighted_scores[gender] = (weighted_scores[gender] / total_weight) * 100

    final_gender = 'Woman' if weighted_scores['Woman'] > weighted_scores['Man'] else 'Man'
    final_confidence = weighted_scores[final_gender]

    return {
        'gender': final_gender,
        'confidence': final_confidence,
        'all_scores': weighted_scores,
        'individual_results': gender_results
    }
def analyze_face_features_advanced(image, selected_features=['age', 'gender', 'emotion', 'race']):
    start_time = time.time()
    try:
        validation_result = validate_image_advanced(image)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error']
            }

        valid_features = ['age', 'gender', 'emotion', 'race']
        features_to_analyze = [f for f in selected_features if f in valid_features]

        if not features_to_analyze:
            return {
                'success': False,
                'error': 'No valid features selected for analysis'
            }

        print(f"Analyzing features: {features_to_analyze}")

        analysis_results = {}
        if 'gender' in features_to_analyze:
            print("Running improved gender analysis...")
            gender_results = analyze_gender_with_multiple_models(image)

            if gender_results:
                consensus_gender = get_consensus_gender(gender_results)

                if consensus_gender:
                    analysis_results['gender'] = {
                        'value': consensus_gender['gender'].lower(),
                        'display': consensus_gender['gender'],
                        'confidence': round(consensus_gender['confidence'], 1),
                        'all_scores': {
                            'man': round(consensus_gender['all_scores'].get('Man', 0), 1),
                            'woman': round(consensus_gender['all_scores'].get('Woman', 0), 1)
                        },
                        'analysis_details': {
                            'predictions_count': len(gender_results),
                            'consensus_method': 'weighted_average'
                        }
                    }

                    print(f"Gender consensus: {consensus_gender['gender']} ({consensus_gender['confidence']:.1f}%)")

                    features_to_analyze = [f for f in features_to_analyze if f != 'gender']

        processed_image = preprocess_image_enhanced(image)

        detector_backends = ['retinaface', 'mtcnn', 'opencv']
        result = None

        if features_to_analyze:
            for backend in detector_backends:
                try:
                    print(f"Trying detector backend: {backend}")
                    result = DeepFace.analyze(
                        img_path=processed_image,
                        actions=features_to_analyze,
                        enforce_detection=True,
                        detector_backend=backend,
                        align=True,
                        silent=True
                    )

                    if isinstance(result, list):
                        if len(result) == 0:
                            continue
                        result = select_best_face(result)

                    if result and len(result) > 0:
                        print(f"Successfully analyzed with {backend} backend")
                        break

                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
                    continue

        if result is None and not analysis_results:
            return {
                'success': False,
                'error': 'Face detection failed with all available methods. Please ensure your face is clearly visible and well-lit.'
            }

        age_predictions = []
        final_age = None

        if 'age' in features_to_analyze:
            primary_age = result.get('age', 0)
            age_predictions.append(primary_age)

            for i in range(2):
                try:
                    variant_image = apply_preprocessing_variant(processed_image, i + 1)

                    variant_result = DeepFace.analyze(
                        img_path=variant_image,
                        actions=['age'],
                        enforce_detection=False,  
                        detector_backend='retinaface', 
                        align=True,
                        silent=True
                    )

                    if isinstance(variant_result, list) and len(variant_result) > 0:
                        variant_result = variant_result[0]

                    if variant_result and 'age' in variant_result:
                        age_predictions.append(variant_result['age'])

                except Exception as e:
                    print(f"Age variant {i+1} failed: {e}")
                    continue

            if len(age_predictions) >= 2:
                weights = [0.6] + [0.4 / (len(age_predictions) - 1)] * (len(age_predictions) - 1)
                final_age = int(np.average(age_predictions, weights=weights))
                print(f"Age predictions: {age_predictions}, Final age: {final_age}")
            else:
                final_age = int(primary_age)

        face_region = {}
        if result:
            face_region = result.get('region', {})
            if face_region:
                face_region = {
                    'x': int(face_region.get('x', 0)),
                    'y': int(face_region.get('y', 0)),
                    'w': int(face_region.get('w', 0)),
                    'h': int(face_region.get('h', 0))
                }

        metadata = {
            'processing_time': round(time.time() - start_time, 2),
            'face_region': face_region,
            'detection_confidence': float(calculate_detection_confidence(result) if result else 85.0),
            'improved_gender_analysis': 'gender' in analysis_results
        }

        if 'age' in features_to_analyze:
            age_value = final_age if final_age is not None else int(float(result['age']))
            age_group = get_simplified_age_group(age_value)
            age_range = get_refined_age_range(age_value)

            confidence = calculate_age_confidence(age_value, age_predictions)

            analysis_results['age'] = {
                'value': age_value,
                'category': age_group['group'],
                'category_emoji': age_group['emoji'],
                'display': f"{age_group['emoji']} {age_value} years old ({age_group['group']})",
                'range': age_range,
                'confidence': confidence,
                'detailed_info': {
                    'exact_age': age_value,
                    'age_group': age_group['group'],
                    'life_stage': age_group['life_stage'],
                    'estimated_range': age_range,
                    'prediction_stability': len(age_predictions),
                    'all_predictions': age_predictions if len(age_predictions) > 1 else None
                }
            }

        if 'age' in features_to_analyze and result:
            age_value = final_age if final_age is not None else int(float(result['age']))
            age_group = get_simplified_age_group(age_value)
            age_range = get_refined_age_range(age_value)

            confidence = calculate_age_confidence(age_value, age_predictions)

            analysis_results['age'] = {
                'value': age_value,
                'category': age_group['group'],
                'category_emoji': age_group['emoji'],
                'display': f"{age_group['emoji']} {age_value} years old ({age_group['group']})",
                'range': age_range,
                'confidence': confidence,
                'detailed_info': {
                    'exact_age': age_value,
                    'age_group': age_group['group'],
                    'life_stage': age_group['life_stage'],
                    'estimated_range': age_range,
                    'prediction_stability': len(age_predictions),
                    'all_predictions': age_predictions if len(age_predictions) > 1 else None
                }
            }

        if 'emotion' in features_to_analyze and result:
            emotion = str(result['dominant_emotion'])
            confidence = float(result['emotion'][emotion])

            all_scores = {}
            for k, v in result['emotion'].items():
                all_scores[str(k)] = round(float(v), 1)

            analysis_results['emotion'] = {
                'value': emotion,
                'display': format_emotion_display(emotion),
                'confidence': round(confidence, 1),
                'all_scores': all_scores
            }

        if 'race' in features_to_analyze and result:
            race = str(result['dominant_race'])
            confidence = float(result['race'][race])

            all_scores = {}
            for k, v in result['race'].items():
                all_scores[str(k)] = round(float(v), 1)

            analysis_results['race'] = {
                'value': race,
                'display': format_race_display(race),
                'confidence': round(confidence, 1),
                'all_scores': all_scores
            }

        validated_results = validate_analysis_results(analysis_results)

        final_result = {
            'success': True,
            'data': validated_results,
            'metadata': metadata
        }

        return final_result

    except Exception as e:
        error_message = str(e)
        print(f"Error in facial analysis: {error_message}")

        if "No face" in error_message or "Face could not be detected" in error_message:
            return {
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible, well-lit, and facing the camera.'
            }
        elif "Invalid image" in error_message or "corrupted" in error_message.lower():
            return {
                'success': False,
                'error': 'Invalid or corrupted image. Please try capturing again.'
            }
        elif "memory" in error_message.lower():
            return {
                'success': False,
                'error': 'Insufficient memory for analysis. Please try again.'
            }
        else:
            return {
                'success': False,
                'error': f'Analysis failed: {error_message}'
            }

def validate_image_advanced(image):
    try:
        if image is None:
            return {'valid': False, 'error': "No image provided"}

        if len(image.shape) < 2:
            return {'valid': False, 'error': "Invalid image format"}

        height, width = image.shape[:2]
        if height < 150 or width < 150:  
            return {'valid': False, 'error': "Image resolution too low for accurate analysis"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)

        if mean_brightness < 30:  
            return {'valid': False, 'error': "Image too dark. Please improve lighting"}
        elif mean_brightness > 225:
            return {'valid': False, 'error': "Image too bright. Please reduce lighting"}

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 15: 
            return {'valid': False, 'error': "Image too blurry. Please hold camera steady"}

        return {'valid': True, 'error': None}

    except Exception as e:
        return {'valid': False, 'error': f"Image validation error: {str(e)}"}

def preprocess_image_enhanced(image):
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()

        if len(processed.shape) == 3:
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)

            enhanced_lab = cv2.merge([l_enhanced, a, b])
            processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        if len(processed.shape) == 3:
            processed = cv2.bilateralFilter(processed, 9, 75, 75)

        height, width = processed.shape[:2]
        if height < 224 or width < 224:
            scale_factor = max(224/height, 224/width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return processed

    except Exception as e:
        print(f"Error in enhanced preprocessing: {e}")
        return preprocess_image_simple(image)

def apply_preprocessing_variant(image, variant_id):
    try:
        if variant_id == 0:
            return image
        elif variant_id == 1:
            adjusted = cv2.convertScaleAbs(image, alpha=1.05, beta=5)
            return adjusted
        elif variant_id == 2:
            adjusted = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
            return adjusted
        else:
            return image
    except:
        return image

def calculate_age_confidence(predicted_age, age_predictions):
    if len(age_predictions) <= 1:
        return 75.0

    std_dev = np.std(age_predictions)

    if std_dev <= 1.5:
        confidence = 95.0
    elif std_dev <= 3:
        confidence = 88.0
    elif std_dev <= 5:
        confidence = 80.0
    elif std_dev <= 7:
        confidence = 72.0
    else:
        confidence = 65.0

    if predicted_age < 18 or predicted_age > 65:
        confidence = max(60.0, confidence - 5.0)

    return round(confidence, 1)

def get_simplified_age_group(age):
    if age <= 12:
        return {
            'group': 'Child',
            'emoji': '🧒',
            'life_stage': 'Childhood'
        }
    elif age <= 19:
        return {
            'group': 'Teen',
            'emoji': '🧑‍🎓',
            'life_stage': 'Adolescence'
        }
    elif age <= 35:
        return {
            'group': 'Young Adult',
            'emoji': '🧑‍💼',
            'life_stage': 'Young Adulthood'
        }
    elif age <= 55:
        return {
            'group': 'Adult',
            'emoji': '👨‍💼',
            'life_stage': 'Middle Adulthood'
        }
    else:
        return {
            'group': 'Senior',
            'emoji': '👴',
            'life_stage': 'Senior Years'
        }

def get_refined_age_range(age):
    if age <= 15:
        margin = 2
    elif age <= 25:
        margin = 3
    elif age <= 40:
        margin = 4
    elif age <= 60:
        margin = 5
    else:
        margin = 6

    min_age = max(0, age - margin)
    max_age = min(120, age + margin)

    return f"{min_age}-{max_age} years"

def preprocess_image_simple(image):
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()
        return processed
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image

def select_best_face(faces_list):
    if len(faces_list) == 1:
        return faces_list[0]

    best_face = faces_list[0]
    max_area = 0

    for face in faces_list:
        if 'region' in face:
            region = face['region']
            area = region.get('w', 0) * region.get('h', 0)
            if area > max_area:
                max_area = area
                best_face = face

    return best_face

def calculate_detection_confidence(result):
    try:
        if 'region' in result:
            region = result['region']
            area = region.get('w', 0) * region.get('h', 0)
            confidence = min(95.0, max(60.0, (area / 10000) * 100))
            return round(confidence, 1)
        return 85.0
    except:
        return 85.0

def format_emotion_display(emotion):
    emotion_map = {
        'angry': 'Angry 😠',
        'disgust': 'Disgusted 🤢',
        'fear': 'Fearful 😨',
        'happy': 'Happy 😊',
        'sad': 'Sad 😢',
        'surprise': 'Surprised 😲',
        'neutral': 'Neutral 😐'
    }
    return emotion_map.get(emotion.lower(), emotion.capitalize())

def format_race_display(race):
    race_map = {
        'asian': 'Asian',
        'indian': 'Indian',
        'black': 'Black',
        'white': 'White',
        'middle eastern': 'Middle Eastern',
        'latino hispanic': 'Latino/Hispanic'
    }
    return race_map.get(race.lower(), race.replace('_', ' ').title())

def validate_analysis_results(analysis_results):
    validated = analysis_results.copy()
    if 'age' in validated:
        age_value = validated['age']['value']

        if age_value < 0:
            validated['age']['value'] = 1
            validated['age']['display'] = f"1 year old (Child)"
        elif age_value > 120:
            validated['age']['value'] = 100
            validated['age']['display'] = f"100 years old (Senior)"

        if validated['age']['confidence'] > 100:
            validated['age']['confidence'] = 95.0
        elif validated['age']['confidence'] < 0:
            validated['age']['confidence'] = 60.0

    for feature in ['gender', 'emotion', 'race']:
        if feature in validated:
            confidence = validated[feature]['confidence']
            if confidence > 100:
                validated[feature]['confidence'] = 95.0
            elif confidence < 0:
                validated[feature]['confidence'] = 50.0

    return validated