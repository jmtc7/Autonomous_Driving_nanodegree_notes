// Auto-generated. Do not edit!

// (in-package simple_arm.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class GoToPositionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.joint_1 = null;
      this.joint_2 = null;
    }
    else {
      if (initObj.hasOwnProperty('joint_1')) {
        this.joint_1 = initObj.joint_1
      }
      else {
        this.joint_1 = 0.0;
      }
      if (initObj.hasOwnProperty('joint_2')) {
        this.joint_2 = initObj.joint_2
      }
      else {
        this.joint_2 = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GoToPositionRequest
    // Serialize message field [joint_1]
    bufferOffset = _serializer.float64(obj.joint_1, buffer, bufferOffset);
    // Serialize message field [joint_2]
    bufferOffset = _serializer.float64(obj.joint_2, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GoToPositionRequest
    let len;
    let data = new GoToPositionRequest(null);
    // Deserialize message field [joint_1]
    data.joint_1 = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [joint_2]
    data.joint_2 = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a service object
    return 'simple_arm/GoToPositionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '243293dc5d540de7ec323fc657126846';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 joint_1
    float64 joint_2
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GoToPositionRequest(null);
    if (msg.joint_1 !== undefined) {
      resolved.joint_1 = msg.joint_1;
    }
    else {
      resolved.joint_1 = 0.0
    }

    if (msg.joint_2 !== undefined) {
      resolved.joint_2 = msg.joint_2;
    }
    else {
      resolved.joint_2 = 0.0
    }

    return resolved;
    }
};

class GoToPositionResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.time_elapsed = null;
    }
    else {
      if (initObj.hasOwnProperty('time_elapsed')) {
        this.time_elapsed = initObj.time_elapsed
      }
      else {
        this.time_elapsed = {secs: 0, nsecs: 0};
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GoToPositionResponse
    // Serialize message field [time_elapsed]
    bufferOffset = _serializer.duration(obj.time_elapsed, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GoToPositionResponse
    let len;
    let data = new GoToPositionResponse(null);
    // Deserialize message field [time_elapsed]
    data.time_elapsed = _deserializer.duration(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'simple_arm/GoToPositionResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5cf2a912daf51cc83cb45e261a19d4f1';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    duration time_elapsed
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GoToPositionResponse(null);
    if (msg.time_elapsed !== undefined) {
      resolved.time_elapsed = msg.time_elapsed;
    }
    else {
      resolved.time_elapsed = {secs: 0, nsecs: 0}
    }

    return resolved;
    }
};

module.exports = {
  Request: GoToPositionRequest,
  Response: GoToPositionResponse,
  md5sum() { return 'fc4e1ffd0bd5c9cc8c021e351562f1a8'; },
  datatype() { return 'simple_arm/GoToPosition'; }
};
