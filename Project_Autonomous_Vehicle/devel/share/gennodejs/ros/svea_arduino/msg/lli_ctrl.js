// Auto-generated. Do not edit!

// (in-package svea_arduino.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class lli_ctrl {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.steering = null;
      this.velocity = null;
      this.trans_diff = null;
      this.ctrl = null;
    }
    else {
      if (initObj.hasOwnProperty('steering')) {
        this.steering = initObj.steering
      }
      else {
        this.steering = 0;
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = 0;
      }
      if (initObj.hasOwnProperty('trans_diff')) {
        this.trans_diff = initObj.trans_diff
      }
      else {
        this.trans_diff = 0;
      }
      if (initObj.hasOwnProperty('ctrl')) {
        this.ctrl = initObj.ctrl
      }
      else {
        this.ctrl = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type lli_ctrl
    // Serialize message field [steering]
    bufferOffset = _serializer.int8(obj.steering, buffer, bufferOffset);
    // Serialize message field [velocity]
    bufferOffset = _serializer.int8(obj.velocity, buffer, bufferOffset);
    // Serialize message field [trans_diff]
    bufferOffset = _serializer.int8(obj.trans_diff, buffer, bufferOffset);
    // Serialize message field [ctrl]
    bufferOffset = _serializer.int8(obj.ctrl, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type lli_ctrl
    let len;
    let data = new lli_ctrl(null);
    // Deserialize message field [steering]
    data.steering = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [velocity]
    data.velocity = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [trans_diff]
    data.trans_diff = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [ctrl]
    data.ctrl = _deserializer.int8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'svea_arduino/lli_ctrl';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f4c1d25e08fe7c24fca84a1ec3ad2a96';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int8 steering
    int8 velocity
    int8 trans_diff
    int8 ctrl
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new lli_ctrl(null);
    if (msg.steering !== undefined) {
      resolved.steering = msg.steering;
    }
    else {
      resolved.steering = 0
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = msg.velocity;
    }
    else {
      resolved.velocity = 0
    }

    if (msg.trans_diff !== undefined) {
      resolved.trans_diff = msg.trans_diff;
    }
    else {
      resolved.trans_diff = 0
    }

    if (msg.ctrl !== undefined) {
      resolved.ctrl = msg.ctrl;
    }
    else {
      resolved.ctrl = 0
    }

    return resolved;
    }
};

module.exports = lli_ctrl;
